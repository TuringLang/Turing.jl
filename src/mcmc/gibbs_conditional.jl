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

supports_gibbs(::GibbsConditional) = true
# The conditional distributions are rebuilt from the conditioned values on every step, so
# nothing sized for the block survives between them.
keeps_linked_layout(::GibbsConditional) = false

"""
    build_values_vnt(model::DynamicPPL.Model)

Build a `VarNamedTuple` of the values of every variable this component conditions on: those
supplied as model arguments, and those Gibbs, the user, or `fix` conditioned.

`merge` is right-biased and replaces a whole key, so an array argument with a `missing`
element -- whose other elements are observations and whose `missing` one Gibbs conditions on
the current draw -- loses its observations to the partially-set conditioned value. Those
elements are put back afterwards, rather than merging leaf by leaf throughout, so that a value
stored under one key stays under one key: `get_cond_dists` sees these keys.

A `missing` in the context is put back the same way. `condition(model; y=missing)` leaves `y`
absent as far as model execution is concerned, so the model uses the argument it was given, and
`get_cond_dists` must see that argument rather than the `missing` the merge would otherwise
leave in its place.
"""
function build_values_vnt(model::DynamicPPL.Model)
    context = model.context
    # Keyword arguments too: `model.args` alone left `f(; y = 2.0)` invisible here.
    args = DynamicPPL.VarNamedTuple(model_argument_values(model))
    vals = merge(args, DynamicPPL.conditioned(context), DynamicPPL.fixed(context))
    for vn in keys(args), leaf in AbstractPPL.varname_leaves(vn, args[vn])
        arg_value = DynamicPPL.getvalue(args, leaf)
        arg_value === missing && continue
        # Restore the argument both when the merge dropped the leaf and when it replaced it
        # with a `missing`, which model execution would have ignored.
        if DynamicPPL.hasvalue(vals, leaf)
            DynamicPPL.getvalue(vals, leaf) === missing || continue
        end
        vals = DynamicPPL.setindex!!(vals, arg_value, leaf)
    end
    return vals
end

@inline _to_varnamedtuple(dists::NamedTuple, ::DynamicPPL.VarNamedTuple) =
    DynamicPPL.VarNamedTuple(dists)
@inline _to_varnamedtuple(dists::DynamicPPL.VarNamedTuple, ::DynamicPPL.VarNamedTuple) =
    dists
function _to_varnamedtuple(dists::AbstractDict{<:VarName}, ::DynamicPPL.VarNamedTuple)
    vnt = DynamicPPL.VarNamedTuple()
    for (vn, dist) in dists
        vnt = DynamicPPL.templated_setindex!!(vnt, dist, vn, _shape_template(dist))
    end
    return vnt
end

"""
    _shape_template(dist)

A container shaped and typed like `dist`, for `templated_setindex!!` to size an array by, or
`NoTemplate()` when the distribution says nothing about a shape. Only its shape and element
type are read, so it is left uninitialised.

The shape has to come from the distribution being written, not from the values the state holds:
those describe the previous step, and another component may have changed the block's dimension
since, in which case templating a two-element result onto a one-element hint throws a
`BoundsError` from inside the setindex.

`NoTemplate()` alone is not enough either. A `VarName` carrying a `Colon` -- `m[:] ~ MvNormal(...)`
-- gives the setindex nothing to infer a size from and it refuses outright. A multivariate or
matrix-variate distribution knows its own size, which is exactly the missing hint; a univariate
one at an indexed `VarName` does not, and does not need it.
"""
_shape_template(::Distributions.UnivariateDistribution) = DynamicPPL.NoTemplate()
_shape_template(d::Distributions.Distribution) = Array{eltype(d)}(undef, size(d))
_shape_template(_) = DynamicPPL.NoTemplate()
function _to_varnamedtuple(dist::Distribution, raw_values::DynamicPPL.VarNamedTuple)
    vns = collect(keys(raw_values))
    # Distinct variables, not distinct keys. One tilde statement can leave several: a
    # `theta[:] ~ MvNormal(...)` site stores `theta[1], theta[2]`, so counting keys refused
    # the very form this method exists to support.
    syms = unique(AbstractPPL.getsym(vn) for vn in vns)
    if length(syms) > 1
        msg = (
            "In GibbsConditional, `get_cond_dists` returned a single distribution," *
            " but multiple variables ($syms) are being sampled. Please return a" *
            " VarNamedTuple mapping variable names to distributions instead."
        )
        throw(ArgumentError(msg))
    end
    top_sym = only(syms)
    # At the variable, not at a leaf, when the leaves are several: the distribution is the
    # conditional for the whole of it, and `_cond_dist_for` resolves each tilde key to it.
    vn = length(vns) == 1 ? only(vns) : VarName{top_sym}()
    template = get(raw_values.data, top_sym, DynamicPPL.NoTemplate())
    return DynamicPPL.templated_setindex!!(DynamicPPL.VarNamedTuple(), dist, vn, template)
end

struct InitFromCondDists{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    cond_dists::V
end
"""
    _cond_dist_for(cond_dists, vn)

The conditional distribution governing the tilde statement at `vn`.

Keyed by subsumption, not by equality. The conditional may be stored at a coarser `VarName`
than the statement uses -- `theta[:] ~ MvNormal(...)` is one distribution for the whole of
`theta`, and the single-distribution form of `get_cond_dists` never sees the tilde key -- so an
exact lookup raised a `MethodError` from inside the distribution.
"""
function _cond_dist_for(cond_dists::DynamicPPL.VarNamedTuple, vn::VarName)
    for key in keys(cond_dists)
        (key == vn || AbstractPPL.subsumes(key, vn)) && return cond_dists[key]
    end
    throw(KeyError(vn))
end

function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, ::Distribution, init_strat::InitFromCondDists
)
    return DynamicPPL.TransformedValue(
        rand(rng, _cond_dist_for(init_strat.cond_dists, vn)), DynamicPPL.NoTransform()
    )
end

# `GibbsConditional` needs the conditioning Gibbs supplies, and returns no transition, so on
# its own it yields neither the right target nor a usable chain. Gibbs is the only caller that
# discards the sample, which is how we tell the two apart.
function error_if_outside_gibbs(discard_sample::Bool)
    discard_sample && return nothing
    return throw(
        ArgumentError(
            "GibbsConditional can only be used as a component of Gibbs. " *
            "Are you trying to use GibbsConditional outside of Gibbs?",
        ),
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::GibbsConditional;
    initial_params,
    discard_sample::Bool=false,
    kwargs...,
)
    error_if_outside_gibbs(discard_sample)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, accs
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::GibbsConditional,
    state::DynamicPPL.OnlyAccsVarInfo;
    discard_sample::Bool=false,
    kwargs...,
)
    error_if_outside_gibbs(discard_sample)
    # Get all the conditioned variable values from the model context. Gibbs conditions the
    # component's non-target variables, so they are in the stack by the time we get here.
    condvals = build_values_vnt(model)
    # `sampler.get_cond_dists(condvals)` could return many things, unfortunately, so we need
    # to handle the different cases.
    #   - just a distribution, in which case we assume there is only one variable being
    #     sampled, and we can just sample from it directly.
    #   - a VarNamedTuple of distributions
    #   - a NamedTuple of distributions
    #   - an AbstractDict mapping VarNames to distributions
    raw_values = DynamicPPL.get_raw_values(state)
    conddists = _to_varnamedtuple(sampler.get_cond_dists(condvals), raw_values)

    init_strategy = InitFromCondDists(conddists)
    _, new_state = DynamicPPL.init!!(
        rng, model, state, init_strategy, DynamicPPL.UnlinkAll()
    )
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, new_state
end

function gibbs_update_state!!(
    ::GibbsConditional,
    state::DynamicPPL.OnlyAccsVarInfo,
    ::DynamicPPL.Model,
    ::DynamicPPL.VarNamedTuple,
)
    # `step` rebuilds the conditional distributions from the model's conditioned values and
    # then overwrites the state with `init!!`, so nothing carried in it needs updating.
    return state
end

# The state carries no layout for the block, so a shape another component changed needs nothing
# special, for the same reason the four-argument form is a no-op.
function gibbs_update_state!!(
    spl::GibbsConditional,
    state::DynamicPPL.OnlyAccsVarInfo,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
    ::ReshapedBlock,
)
    return gibbs_update_state!!(spl, state, model, global_vals)
end
