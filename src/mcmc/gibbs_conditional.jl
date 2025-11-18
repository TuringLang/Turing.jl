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

Here `get_cond_dists(c::Dict{<:VarName})` should be a function that takes a `Dict` mapping
the conditioned variables (anything other than `var1` and `var2`) to their values, and
returns the conditional posterior distributions for `var1` and `var2`. You may, of course,
have any number of variables being sampled as a block in this manner, we only use two as an
example. The return value of `get_cond_dists` should be one of the following:
- A single `Distribution`, if only one variable is being sampled.
- An `AbstractDict{<:VarName,<:Distribution}` that maps the variables being sampled to their
  conditional posteriors E.g. `Dict(@varname(var1) => dist1, @varname(var2) => dist2)`.
- A `NamedTuple` of `Distribution`s, which is like the `AbstractDict` case but can be used
  if all the variable names are single `Symbol`s, and may be more performant. E.g.
  `(; var1=dist1, var2=dist2)`.

# Examples

```julia
# Define a model
@model function inverse_gdemo(x)
    precision ~ Gamma(2, inv(3))
    std = sqrt(1 / precision)
    m ~ Normal(0, std)
    for i in eachindex(x)
        x[i] ~ Normal(m, std)
    end
end

# Define analytical conditionals
function cond_precision(c)
    a = 2.0
    b = 3.0
    m = c[@varname(m)]
    x = c[@varname(x)]
    n = length(x)
    a_new = a + (n + 1) / 2
    b_new = b + sum(abs2, x .- m) / 2 + m^2 / 2
    return Gamma(a_new, 1 / b_new)
end

function cond_m(c)
    precision = c[@varname(precision)]
    x = c[@varname(x)]
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
    build_variable_dict(model::DynamicPPL.Model)

Traverse the context stack of `model` and build a `Dict` of all the variable values that are
set in GibbsContext, ConditionContext, or FixedContext.
"""
function build_variable_dict(model::DynamicPPL.Model)
    context = model.context
    cond_nt = DynamicPPL.conditioned(context)
    fixed_nt = DynamicPPL.fixed(context)
    # TODO(mhauru) Can we avoid invlinking all the time? Note that this causes a model
    # evaluation, which may be expensive.
    global_vi = DynamicPPL.invlink(get_gibbs_global_varinfo(context), model)
    return merge(
        DynamicPPL.values_as(global_vi, Dict),
        Dict(
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(cond_nt))...,
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(fixed_nt))...,
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(model.args))...,
        ),
    )
end

function get_gibbs_global_varinfo(context::DynamicPPL.AbstractContext)
    return if context isa GibbsContext
        get_global_varinfo(context)
    elseif DynamicPPL.NodeTrait(context) isa DynamicPPL.IsParent
        get_gibbs_global_varinfo(DynamicPPL.childcontext(context))
    else
        msg = """No GibbsContext found in context stack. Are you trying to use \
            GibbsConditional outside of Gibbs?
            """
        throw(ArgumentError(msg))
    end
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
    condvals = build_variable_dict(model)
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
