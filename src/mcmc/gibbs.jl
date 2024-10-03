# Basically like a `DynamicPPL.FixedContext` but
# 1. Hijacks the tilde pipeline to fix variables.
# 2. Computes the log-probability of the fixed variables.
#
# Purpose: avoid triggering resampling of variables we're conditioning on.
# - Using standard `DynamicPPL.condition` results in conditioned variables being treated
#   as observations in the truest sense, i.e. we hit `DynamicPPL.tilde_observe`.
# - But `observe` is overloaded by some samplers, e.g. `CSMC`, which can lead to
#   undesirable behavior, e.g. `CSMC` triggering a resampling for every conditioned variable
#   rather than only for the "true" observations.
# - `GibbsContext` allows us to perform conditioning while still hit the `assume` pipeline
#   rather than the `observe` pipeline for the conditioned variables.
struct GibbsContext{Values,Ctx<:DynamicPPL.AbstractContext} <: DynamicPPL.AbstractContext
    values::Values
    context::Ctx
end

Gibbscontext(values) = GibbsContext(values, DynamicPPL.DefaultContext())

DynamicPPL.NodeTrait(::GibbsContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::GibbsContext) = context.context
function DynamicPPL.setchildcontext(context::GibbsContext, childcontext)
    return GibbsContext(context.values, childcontext)
end

# has and get
function has_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.hasvalue(context.values, vn)
end
function has_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return all(Base.Fix1(has_conditioned_gibbs, context), vns)
end

function get_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.getvalue(context.values, vn)
end
function get_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(get_conditioned_gibbs, context), vns)
end

# Tilde pipeline
function DynamicPPL.tilde_assume(context::GibbsContext, right, vn, vi)
    # Short-circuits the tilde assume if `vn` is present in `context`.
    if has_conditioned_gibbs(context, vn)
        value = get_conditioned_gibbs(context, vn)
        return value, logpdf(right, value), vi
    end

    # Otherwise, falls back to the default behavior.
    return DynamicPPL.tilde_assume(DynamicPPL.childcontext(context), right, vn, vi)
end

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, vn, vi
)
    # Short-circuits the tilde assume if `vn` is present in `context`.
    if has_conditioned_gibbs(context, vn)
        value = get_conditioned_gibbs(context, vn)
        return value, logpdf(right, value), vi
    end

    # Otherwise, falls back to the default behavior.
    return DynamicPPL.tilde_assume(
        rng, DynamicPPL.childcontext(context), sampler, right, vn, vi
    )
end

# Some utility methods for handling the `logpdf` computations in dot-tilde the pipeline.
make_broadcastable(x) = x
make_broadcastable(dist::Distribution) = tuple(dist)

# Need the following two methods to properly support broadcasting over columns.
broadcast_logpdf(dist, x) = sum(logpdf.(make_broadcastable(dist), x))
function broadcast_logpdf(dist::MultivariateDistribution, x::AbstractMatrix)
    return loglikelihood(dist, x)
end

# Needed to support broadcasting over columns for `MultivariateDistribution`s.
reconstruct_getvalue(dist, x) = x
function reconstruct_getvalue(
    dist::MultivariateDistribution, x::AbstractVector{<:AbstractVector{<:Real}}
)
    return reduce(hcat, x[2:end]; init=x[1])
end

function DynamicPPL.dot_tilde_assume(context::GibbsContext, right, left, vns, vi)
    # Short-circuits the tilde assume if `vn` is present in `context`.
    if has_conditioned_gibbs(context, vns)
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        return value, broadcast_logpdf(right, value), vi
    end

    # Otherwise, falls back to the default behavior.
    return DynamicPPL.dot_tilde_assume(
        DynamicPPL.childcontext(context), right, left, vns, vi
    )
end

function DynamicPPL.dot_tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, left, vns, vi
)
    # Short-circuits the tilde assume if `vn` is present in `context`.
    if has_conditioned_gibbs(context, vns)
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        return value, broadcast_logpdf(right, value), vi
    end

    # Otherwise, falls back to the default behavior.
    return DynamicPPL.dot_tilde_assume(
        rng, DynamicPPL.childcontext(context), sampler, right, left, vns, vi
    )
end

"""
    preferred_value_type(varinfo::DynamicPPL.AbstractVarInfo)

Returns the preferred value type for a variable with the given `varinfo`.
"""
preferred_value_type(::DynamicPPL.AbstractVarInfo) = DynamicPPL.OrderedDict
preferred_value_type(::DynamicPPL.SimpleVarInfo{<:NamedTuple}) = NamedTuple
function preferred_value_type(varinfo::DynamicPPL.TypedVarInfo)
    # We can only do this in the scenario where all the varnames are `Accessors.IdentityLens`.
    namedtuple_compatible = all(varinfo.metadata) do md
        eltype(md.vns) <: VarName{<:Any,typeof(identity)}
    end
    return namedtuple_compatible ? NamedTuple : DynamicPPL.OrderedDict
end

"""
    condition_gibbs(context::DynamicPPL.AbstractContext, values::Union{NamedTuple,AbstractDict}...)

Return a `GibbsContext` with the given values treated as conditioned.

# Arguments
- `context::DynamicPPL.AbstractContext`: The context to condition.
- `values::Union{NamedTuple,AbstractDict}...`: The values to condition on.
    If multiple values are provided, we recursively condition on each of them.
"""
condition_gibbs(context::DynamicPPL.AbstractContext) = context
# For `NamedTuple` and `AbstractDict` we just construct the context.
function condition_gibbs(
    context::DynamicPPL.AbstractContext, values::Union{NamedTuple,AbstractDict}
)
    return GibbsContext(values, context)
end
# If we get more than one argument, we just recurse.
function condition_gibbs(context::DynamicPPL.AbstractContext, value, values...)
    return condition_gibbs(condition_gibbs(context, value), values...)
end

# For `DynamicPPL.AbstractVarInfo` we just extract the values.
"""
    condition_gibbs(context::DynamicPPL.AbstractContext, varinfos::DynamicPPL.AbstractVarInfo...)

Return a `GibbsContext` with the values extracted from the given `varinfos` treated as conditioned.
"""
function condition_gibbs(
    context::DynamicPPL.AbstractContext, varinfo::DynamicPPL.AbstractVarInfo
)
    return condition_gibbs(
        context, DynamicPPL.values_as(varinfo, preferred_value_type(varinfo))
    )
end
function condition_gibbs(
    context::DynamicPPL.AbstractContext,
    varinfo::DynamicPPL.AbstractVarInfo,
    varinfos::DynamicPPL.AbstractVarInfo...,
)
    return condition_gibbs(condition_gibbs(context, varinfo), varinfos...)
end
# Allow calling this on a `DynamicPPL.Model` directly.
function condition_gibbs(model::DynamicPPL.Model, values...)
    return DynamicPPL.contextualize(model, condition_gibbs(model.context, values...))
end

"""
    make_conditional_model(model, varinfo, varinfos)

Construct a conditional model from `model` conditioned `varinfos`, excluding `varinfo` if present.

# Examples
```julia-repl
julia> model = DynamicPPL.TestUtils.demo_assume_dot_observe();

julia> # A separate varinfo for each variable in `model`.
       varinfos = (DynamicPPL.SimpleVarInfo(s=1.0), DynamicPPL.SimpleVarInfo(m=10.0));

julia> # The varinfo we want to NOT condition on.
       target_varinfo = first(varinfos);

julia> # Results in a model with only `m` conditioned.
       conditioned_model = make_conditional(model, target_varinfo, varinfos);

julia> result = conditioned_model();

julia> result.m == 10.0  # we conditioned on varinfo with `m = 10.0`
true

julia> result.s != 1.0  # we did NOT want to condition on varinfo with `s = 1.0`
true
```
"""
function make_conditional(
    model::DynamicPPL.Model, target_varinfo::DynamicPPL.AbstractVarInfo, varinfos
)
    # TODO: Check if this is known at compile-time if `varinfos isa Tuple`.
    return condition_gibbs(model, filter(Base.Fix1(!==, target_varinfo), varinfos)...)
end
# Assumes the ones given are the ones to condition on.
function make_conditional(model::DynamicPPL.Model, varinfos)
    return condition_gibbs(model, varinfos...)
end

# HACK: Allows us to support either passing in an implementation of `AbstractMCMC.AbstractSampler`
# or an `AbstractInferenceAlgorithm`.
wrap_algorithm_maybe(x) = x
wrap_algorithm_maybe(x::InferenceAlgorithm) = DynamicPPL.Sampler(x)

"""
    gibbs_state(model, sampler, state, varinfo)

Return an updated state, taking into account the variables sampled by other Gibbs components.

# Arguments
- `model`: model targeted by the Gibbs sampler.
- `sampler`: the sampler for this Gibbs component.
- `state`: the state of `sampler` computed in the previous iteration.
- `varinfo`: the variables, including the ones sampled by other Gibbs components.
"""
gibbs_state(model, sampler, state::AbstractVarInfo, varinfo::AbstractVarInfo) = varinfo
function gibbs_state(model, sampler, state::PGState, varinfo::AbstractVarInfo)
    return PGState(varinfo, state.rng)
end

# Update state in Gibbs sampling
function gibbs_state(
    model::Model, spl::Sampler{<:Hamiltonian}, state::HMCState, varinfo::AbstractVarInfo
)
    # Update hamiltonian
    θ_old = varinfo[spl]
    hamiltonian = get_hamiltonian(model, spl, varinfo, state, length(θ_old))

    # TODO: Avoid mutation
    resize!(state.z.θ, length(θ_old))
    state.z.θ .= θ_old
    z = state.z

    return HMCState(varinfo, state.i, state.kernel, hamiltonian, z, state.adaptor)
end

"""
    Gibbs

A type representing a Gibbs sampler.

# Fields
$(TYPEDFIELDS)
"""
struct Gibbs{V,A} <: InferenceAlgorithm
    "varnames representing variables for each sampler"
    varnames::V
    "samplers for each entry in `varnames`"
    samplers::A
end

# NamedTuple
Gibbs(; algs...) = Gibbs(NamedTuple(algs))
function Gibbs(algs::NamedTuple)
    return Gibbs(
        map(s -> VarName{s}(), keys(algs)), map(wrap_algorithm_maybe, values(algs))
    )
end

# AbstractDict
function Gibbs(algs::AbstractDict)
    return Gibbs(collect(keys(algs)), map(wrap_algorithm_maybe, values(algs)))
end
function Gibbs(algs::Pair...)
    return Gibbs(map(first, algs), map(wrap_algorithm_maybe, map(last, algs)))
end

# The below two constructors only provide backwards compatibility with the constructor of
# the old Gibbs sampler. They are deprecated and will be removed in the future.
function Gibbs(algs::InferenceAlgorithm...)
    varnames = map(algs) do alg
        space = getspace(alg)
        if (space isa VarName)
            space
        elseif (space isa Symbol)
            VarName{space}()
        else
            tuple((s isa Symbol ? VarName{s}() : s for s in space)...)
        end
    end
    msg = (
        "Specifying which sampler to use with which variable using syntax like " *
        "`Gibbs(NUTS(:x), MH(:y))` is deprecated and will be removed in the future. " *
        "Please use `Gibbs(; x=NUTS(), y=MH())` instead. If you want different iteration " *
        "counts for different subsamplers, use e.g. " *
        "`Gibbs(@varname(x) => NUTS(), @varname(x) => NUTS(), @varname(y) => MH())`"
    )
    Base.depwarn(msg, :Gibbs)
    return Gibbs(varnames, map(wrap_algorithm_maybe, algs))
end

function Gibbs(algs_with_iters::Tuple{<:InferenceAlgorithm,Int}...)
    algs = Iterators.map(first, algs_with_iters)
    iters = Iterators.map(last, algs_with_iters)
    algs_duplicated = Iterators.flatten((
        Iterators.repeated(alg, iter) for (alg, iter) in zip(algs, iters)
    ))
    # This calls the other deprecated constructor from above, hence no need for a depwarn
    # here.
    return Gibbs(algs_duplicated...)
end

# TODO: Remove when no longer needed.
DynamicPPL.getspace(::Gibbs) = ()

struct GibbsState{V<:DynamicPPL.AbstractVarInfo,S}
    vi::V
    states::S
end

_maybevec(x) = vec(x)  # assume it's iterable
_maybevec(x::Tuple) = [x...]
_maybevec(x::VarName) = [x]

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs},
    vi_base::DynamicPPL.AbstractVarInfo;
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers

    # 1. Run the model once to get the varnames present + initial values to condition on.
    vi_base = DynamicPPL.VarInfo(rng, model)

    # Simple way of setting the initial parameters: set them in the `vi_base`
    # if they are given so they propagate to the subset varinfos used by each sampler.
    if initial_params !== nothing
        vi_base = DynamicPPL.unflatten(vi_base, initial_params)
    end

    # Create the varinfos for each sampler.
    varinfos = map(Base.Fix1(DynamicPPL.subset, vi_base) ∘ _maybevec, varnames)
    initial_params_all = if initial_params === nothing
        fill(nothing, length(varnames))
    else
        # Extract from the `vi_base`, which should have the values set correctly from above.
        map(vi -> vi[:], varinfos)
    end

    # 2. Construct a varinfo for every vn + sampler combo.
    states_and_varinfos = map(
        samplers, varinfos, initial_params_all
    ) do sampler_local, varinfo_local, initial_params_local
        # Construct the conditional model.
        model_local = make_conditional(model, varinfo_local, varinfos)

        # Take initial step.
        new_state_local = last(
            AbstractMCMC.step(
                rng,
                model_local,
                sampler_local;
                # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
                # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
                initial_params=initial_params_local,
                kwargs...,
            ),
        )

        # Return the new state and the invlinked `varinfo`.
        vi_local_state = varinfo(new_state_local)
        vi_local_state_linked = if DynamicPPL.istrans(vi_local_state)
            DynamicPPL.invlink(vi_local_state, sampler_local, model_local)
        else
            vi_local_state
        end
        return (new_state_local, vi_local_state_linked)
    end

    states = map(first, states_and_varinfos)
    varinfos = map(last, states_and_varinfos)

    # Update the base varinfo from the first varinfo and replace it.
    varinfos_new = DynamicPPL.setindex!!(varinfos, merge(vi_base, first(varinfos)), 1)
    # Merge the updated initial varinfo with the rest of the varinfos + update the logp.
    vi = DynamicPPL.setlogp!!(
        reduce(merge, varinfos_new), DynamicPPL.getlogp(last(varinfos))
    )

    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs},
    state::GibbsState;
    kwargs...,
)
    alg = spl.alg
    samplers = alg.samplers
    states = state.states
    varinfos = map(varinfo, state.states)
    @assert length(samplers) == length(state.states)

    # TODO: move this into a recursive function so we can unroll when reasonable?
    for index in 1:length(samplers)
        # Take the inner step.
        new_state_local, new_varinfo_local = gibbs_step_inner(
            rng, model, samplers, states, varinfos, index; kwargs...
        )

        # Update the `states` and `varinfos`.
        states = Accessors.setindex(states, new_state_local, index)
        varinfos = Accessors.setindex(varinfos, new_varinfo_local, index)
    end

    # Combine the resulting varinfo objects.
    # The last varinfo holds the correctly computed logp.
    vi_base = state.vi

    # Update the base varinfo from the first varinfo and replace it.
    varinfos_new = DynamicPPL.setindex!!(
        varinfos, merge(vi_base, first(varinfos)), firstindex(varinfos)
    )
    # Merge the updated initial varinfo with the rest of the varinfos + update the logp.
    vi = DynamicPPL.setlogp!!(
        reduce(merge, varinfos_new), DynamicPPL.getlogp(last(varinfos))
    )

    return Transition(model, vi), GibbsState(vi, states)
end

# TODO: Remove this once we've done away with the selector functionality in DynamicPPL.
function make_rerun_sampler(model::DynamicPPL.Model, sampler::DynamicPPL.Sampler)
    # NOTE: This is different from the implementation used in the old `Gibbs` sampler, where we specifically provide
    # a `gid`. Here, because `model` only contains random variables to be sampled by `sampler`, we just use the exact
    # same `selector` as before but now with `rerun` set to `true` if needed.
    return Accessors.@set sampler.selector.rerun = true
end

# Interface we need a sampler to implement to work as a component in a Gibbs sampler.
"""
    gibbs_requires_recompute_logprob(model_dst, sampler_dst, sampler_src, state_dst, state_src)

Check if the log-probability of the destination model needs to be recomputed.

Defaults to `true`
"""
function gibbs_requires_recompute_logprob(
    model_dst, sampler_dst, sampler_src, state_dst, state_src
)
    return true
end

# TODO: Remove `rng`?
function recompute_logprob!!(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::DynamicPPL.Sampler, state
)
    vi = varinfo(state)
    # NOTE: Need to do this because some samplers might need some other quantity than the log-joint,
    # e.g. log-likelihood in the scenario of `ESS`.
    # NOTE: Need to update `sampler` too because the `gid` might change in the re-run of the model.
    sampler_rerun = make_rerun_sampler(model, sampler)
    # NOTE: If we hit `DynamicPPL.maybe_invlink_before_eval!!`, then this will result in a `invlink`ed
    # `varinfo`, even if `varinfo` was linked.
    vi_new = last(
        DynamicPPL.evaluate!!(
            model,
            vi,
            # TODO: Check if it's safe to drop the `rng` argument, i.e. just use default RNG.
            DynamicPPL.SamplingContext(rng, sampler_rerun),
        )
    )
    # Update the state we're about to use if need be.
    # NOTE: If the sampler requires a linked varinfo, this should be done in `gibbs_state`.
    return gibbs_state(model, sampler, state, vi_new)
end

function gibbs_step_inner(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    samplers,
    states,
    varinfos,
    index;
    kwargs...,
)
    # Needs to do a a few things.
    sampler_local = samplers[index]
    state_local = states[index]
    varinfo_local = varinfos[index]

    # Make sure that all `varinfos` are linked.
    varinfos_invlinked = map(varinfos) do vi
        # NOTE: This is immutable linking!
        # TODO: Do we need the `istrans` check here or should we just always use `invlink`?
        # FIXME: Suffers from https://github.com/TuringLang/Turing.jl/issues/2195
        DynamicPPL.istrans(vi) ? DynamicPPL.invlink(vi, model) : vi
    end
    varinfo_local_invlinked = varinfos_invlinked[index]

    # 1. Create conditional model.
    # Construct the conditional model.
    # NOTE: Here it's crucial that all the `varinfos` are in the constrained space,
    # otherwise we're conditioning on values which are not in the support of the
    # distributions.
    model_local = make_conditional(model, varinfo_local_invlinked, varinfos_invlinked)

    # Extract the previous sampler and state.
    sampler_previous = samplers[index == 1 ? length(samplers) : index - 1]
    state_previous = states[index == 1 ? length(states) : index - 1]

    # 1. Re-run the sampler if needed.
    if gibbs_requires_recompute_logprob(
        model_local, sampler_local, sampler_previous, state_local, state_previous
    )
        state_local = recompute_logprob!!(rng, model_local, sampler_local, state_local)
    end

    # 2. Take step with local sampler.
    new_state_local = last(
        AbstractMCMC.step(rng, model_local, sampler_local, state_local; kwargs...)
    )

    # 3. Extract the new varinfo.
    # Return the resulting state and invlinked `varinfo`.
    varinfo_local_state = varinfo(new_state_local)
    varinfo_local_state_invlinked = if DynamicPPL.istrans(varinfo_local_state)
        DynamicPPL.invlink(varinfo_local_state, sampler_local, model_local)
    else
        varinfo_local_state
    end

    # TODO: alternatively, we can return `states_new, varinfos_new, index_new`
    return (new_state_local, varinfo_local_state_invlinked)
end
