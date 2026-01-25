"""
    GibbsConditional(get_cond_dists)

A Gibbs component sampler that samples variables according to user-provided analytical
conditional posterior distributions.

When using Gibbs sampling, sometimes one may know the analytical form of the posterior for
a given variable, given the conditioned values of the other variables. In such cases one can
use `GibbsConditional` as a component sampler to to sample from these known conditionals
directly, avoiding any MCMC methods. One does so with

```julia
sampler = Gibbs(
    (@varname(var1), @varname(var2)) => GibbsConditional(get_cond_dists),
    other samplers go here...
)
```

Here `get_cond_dists(vnt::VarNamedTuple)` should be a function that takes a `VarNamedTuple`
that contains the values of all other variables (apart from `var1` and `var2`), and returns
the conditional posterior distributions for `var1` and `var2`. `VarNamedTuple`s behave very
similarly to `Dict{VarName,Any}`s, but are more efficient and more general: you can obtain
values simply by using, e.g. `vnt[@varname(var3)]`.

You may, of course, have any number of variables being sampled as a block in this manner, we
only use two as an example.

The return value of `get_cond_dists` should be one of the following:

- A single `Distribution`, if only one variable is being sampled.
- An `AbstractDict{<:VarName,<:Distribution}` that maps the variables being sampled to their
  conditional posteriors E.g. `Dict(@varname(var1) => dist1, @varname(var2) => dist2)`.
- A `NamedTuple` of `Distribution`s, which is like the `AbstractDict` case but can be used
  if all the variable names are single `Symbol`s, and may be more performant, e.g.:
  `(; var1=dist1, var2=dist2)`.

# Examples

```julia
using Turing

# Define a model
@model function inverse_gdemo(x)
    precision ~ Gamma(2, inv(3))
    std = sqrt(1 / precision)
    m ~ Normal(0, std)
    for i in eachindex(x)
        x[i] ~ Normal(m, std)
    end
end

# Define analytical conditionals. See
# https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
function cond_precision(vnt)
    a = 2.0
    b = 3.0
    m = vnt[@varname(m)]
    x = vnt[@varname(x)]
    n = length(x)
    a_new = a + (n + 1) / 2
    b_new = b + sum(abs2, x .- m) / 2 + m^2 / 2
    return Gamma(a_new, 1 / b_new)
end

function cond_m(vnt)
    precision = vnt[@varname(precision)]
    x = vnt[@varname(x)]
    n = length(x)
    m_mean = sum(x) / (n + 1)
    m_var = 1 / (precision * (n + 1))
    return Normal(m_mean, sqrt(m_var))
end

# Sample using GibbsConditional
model = inverse_gdemo([1.0, 2.0, 3.0])
chain = sample(model, Gibbs(
    :precision => GibbsConditional(cond_precision),
    :m => GibbsConditional(cond_m)
), 1000)
```
"""
struct GibbsConditional{C} <: AbstractSampler
    get_cond_dists::C
end

isgibbscomponent(::GibbsConditional) = true

"""
    build_values_vnt(model::DynamicPPL.Model)

Traverse the context stack of `model` and build a `VarNamedTuple` of all the variable values
that are set in GibbsContext, ConditionContext, or FixedContext.
"""
function build_values_vnt(model::DynamicPPL.Model)
    context = model.context
    # TODO(penelopeysm): Use VNTs for ConditionContext and FixedContext
    cond_vals = DynamicPPL.to_varname_dict(DynamicPPL.conditioned(context))
    fixed_vals = DynamicPPL.to_varname_dict(DynamicPPL.fixed(context))
    arg_vals = DynamicPPL.to_varname_dict(model.args)
    # Need to get the invlinked values as a VNT
    vi = deepcopy(get_gibbs_global_varinfo(context))
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    # need to remove the Gibbs conditioning so that we can get all variables in the VarInfo
    defmodel = replace_gibbs_context(model)
    _, vi = DynamicPPL.evaluate!!(defmodel, vi)
    global_vals = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
    # Merge them.
    # TODO(penelopeysm): We don't have templating information here. This could be fixed if
    # we used VNTs everywhere -- in which case we can just merge the VNTs. Although we
    # might have to be careful if a conditioned VNT has no templates and we attempt to
    # merge into one that does......
    for (vn, val) in [pairs(cond_vals)..., pairs(fixed_vals)..., pairs(arg_vals)...]
        global_vals = BangBang.setindex!!(global_vals, val, vn)
    end
    return global_vals
end

replace_gibbs_context(::GibbsContext) = DefaultContext()
replace_gibbs_context(::DynamicPPL.AbstractContext) = DefaultContext()
function replace_gibbs_context(c::DynamicPPL.AbstractParentContext)
    return DynamicPPL.setchildcontext(c, replace_gibbs_context(DynamicPPL.childcontext(c)))
end
function replace_gibbs_context(m::DynamicPPL.Model)
    return DynamicPPL.contextualize(m, replace_gibbs_context(m.context))
end

function get_gibbs_global_varinfo(context::GibbsContext)
    return get_global_varinfo(context)
end
function get_gibbs_global_varinfo(context::DynamicPPL.AbstractParentContext)
    return get_gibbs_global_varinfo(DynamicPPL.childcontext(context))
end
function get_gibbs_global_varinfo(::DynamicPPL.AbstractContext)
    msg = """No GibbsContext found in context stack. Are you trying to use \
        GibbsConditional outside of Gibbs?
        """
    throw(ArgumentError(msg))
end

function initialstep(
    ::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::GibbsConditional,
    vi::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    state = DynamicPPL.is_transformed(vi) ? DynamicPPL.invlink(vi, model) : vi
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::GibbsConditional,
    state::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    # Get all the conditioned variable values from the model context. This is assumed to
    # include a GibbsContext as part of the context stack.
    condvals = build_values_vnt(model)
    conddists = sampler.get_cond_dists(condvals)

    # We support three different kinds of return values for `sample.get_cond_dists`, to make
    # life easier for the user.
    if conddists isa AbstractDict
        for (vn, dist) in conddists
            state = setindex!!(state, rand(rng, dist), vn)
        end
    elseif conddists isa NamedTuple
        for (vn_sym, dist) in pairs(conddists)
            vn = VarName{vn_sym}()
            state = setindex!!(state, rand(rng, dist), vn)
        end
    else
        # Single variable case
        vn = only(keys(state))
        state = setindex!!(state, rand(rng, conddists), vn)
    end

    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, state
end

function setparams_varinfo!!(
    ::DynamicPPL.Model, ::GibbsConditional, ::Any, params::DynamicPPL.AbstractVarInfo
)
    return params
end
