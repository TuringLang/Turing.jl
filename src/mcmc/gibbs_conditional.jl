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
the conditional posterior distributions for `var1` and `var2`.

`VarNamedTuple`s behave very similarly to `Dict{VarName,Any}`s, but are more efficient and
more general: you can obtain values simply by using, e.g. `vnt[@varname(var3)]`. See
https://turinglang.org/docs/usage/varnamedtuple/ for more details on `VarNamedTuple`s.

You may, of course, have any number of variables being sampled as a block in this manner, we
only use two as an example.

The return value of `get_cond_dists(vnt)` should be one of the following:

- A single `Distribution`, if only one variable is being sampled.
- A `VarNamedTuple` of `Distribution`s, which represents a mapping from variable names to their
  conditional posteriors. Please see the documentation linked above for information on how to
  construct `VarNamedTuple`s.

For convenience, we also allow the following return values (which are internally converted into
a `VarNamedTuple`):

- A `NamedTuple` of `Distribution`s, which is like the `AbstractDict` case but can be used
  if all the variable names are single `Symbol`s, e.g.: `(; var1=dist1, var2=dist2)`.
- An `AbstractDict{<:VarName,<:Distribution}` that maps the variables being sampled to their
  conditional posteriors E.g. `Dict(@varname(var1) => dist1, @varname(var2) => dist2)`.

Note that the `AbstractDict` case is likely to incur a performance penalty; we recommend using
`VarNamedTuple`s directly.

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
    cond_vals = DynamicPPL.conditioned(context)
    fixed_vals = DynamicPPL.fixed(context)
    # model.args is a NamedTuple
    arg_vals = DynamicPPL.VarNamedTuple(model.args)
    # Extract values from the GibbsContext itself, as a VNT.
    init_strat = DynamicPPL.InitFromParams(
        get_gibbs_global_varinfo(context).values, nothing
    )
    oavi = DynamicPPL.OnlyAccsVarInfo((DynamicPPL.RawValueAccumulator(false),))
    # We need to remove the Gibbs conditioning so that we can get all variables in the
    # accumulator (otherwise those that are conditioned on in `model` will not be included).
    defmodel = replace_gibbs_context(model)
    _, oavi = DynamicPPL.init!!(defmodel, oavi, init_strat, DynamicPPL.UnlinkAll())
    global_vals = DynamicPPL.get_raw_values(oavi)
    # Merge them.
    return merge(global_vals, cond_vals, fixed_vals, arg_vals)
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

function Turing.Inference.initialstep(
    ::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::GibbsConditional,
    vi::DynamicPPL.VarInfo;
    kwargs...,
)
    state = DynamicPPL.is_transformed(vi) ? DynamicPPL.invlink(vi, model) : vi
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, state
end

@inline _to_varnamedtuple(dists::NamedTuple, ::DynamicPPL.VarInfo) =
    DynamicPPL.VarNamedTuple(dists)
@inline _to_varnamedtuple(dists::DynamicPPL.VarNamedTuple, ::DynamicPPL.VarInfo) = dists
function _to_varnamedtuple(dists::AbstractDict{<:VarName}, state::DynamicPPL.VarInfo)
    template_vnt = state.values
    vnt = DynamicPPL.VarNamedTuple()
    for (vn, dist) in dists
        top_sym = AbstractPPL.getsym(vn)
        template = get(template_vnt.data, top_sym, DynamicPPL.NoTemplate())
        vnt = DynamicPPL.templated_setindex!!(vnt, dist, vn, template)
    end
    return vnt
end
function _to_varnamedtuple(dist::Distribution, state::DynamicPPL.VarInfo)
    vns = keys(state)
    if length(vns) > 1
        msg = (
            "In GibbsConditional, `get_cond_dists` returned a single distribution," *
            " but multiple variables ($vns) are being sampled. Please return a" *
            " VarNamedTuple mapping variable names to distributions instead."
        )
        throw(ArgumentError(msg))
    end
    vn = only(vns)
    top_sym = AbstractPPL.getsym(vn)
    template = get(state.values.data, top_sym, DynamicPPL.NoTemplate())
    return DynamicPPL.templated_setindex!!(DynamicPPL.VarNamedTuple(), dist, vn, template)
end

struct InitFromCondDists{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    cond_dists::V
end
function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, ::Distribution, init_strat::InitFromCondDists
)
    return DynamicPPL.UntransformedValue(rand(rng, init_strat.cond_dists[vn]))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::GibbsConditional,
    state::DynamicPPL.VarInfo;
    kwargs...,
)
    # Get all the conditioned variable values from the model context. This is assumed to
    # include a GibbsContext as part of the context stack.
    condvals = build_values_vnt(model)
    # `sampler.get_cond_dists(condvals)` could return many things, unfortunately, so we need
    # to handle the different cases.
    #   - just a distribution, in which case we assume there is only one variable being
    #     sampled, and we can just sample from it directly.
    #   - a VarNamedTuple of distributions
    #   - a NamedTuple of distributions
    #   - an AbstractDict mapping VarNames to distributions
    conddists = _to_varnamedtuple(sampler.get_cond_dists(condvals), state)

    init_strategy = InitFromCondDists(conddists)
    _, new_state = DynamicPPL.init!!(rng, model, state, init_strategy)
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, new_state
end

function setparams_varinfo!!(
    ::DynamicPPL.Model, ::GibbsConditional, ::Any, params::DynamicPPL.VarInfo
)
    return params
end
