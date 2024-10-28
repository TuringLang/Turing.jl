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
struct GibbsContext{VNs,GVI<:Ref{<:AbstractVarInfo},Ctx<:DynamicPPL.AbstractContext} <:
       DynamicPPL.AbstractContext
    target_varnames::VNs
    global_varinfo::GVI
    context::Ctx
end

function GibbsContext(target_varnames, global_varinfo)
    return GibbsContext(target_varnames, global_varinfo, DynamicPPL.DefaultContext())
end

DynamicPPL.NodeTrait(::GibbsContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::GibbsContext) = context.context
function DynamicPPL.setchildcontext(context::GibbsContext, childcontext)
    return GibbsContext(
        context.target_varnames, Ref(context.global_varinfo[]), childcontext
    )
end

# has and get
function has_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.haskey(context.global_varinfo[], vn)
end
function has_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return all(Base.Fix1(has_conditioned_gibbs, context), vns)
end

function get_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return context.global_varinfo[][vn]
end
function get_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(get_conditioned_gibbs, context), vns)
end

function is_target_varname(context::GibbsContext, vn::VarName)
    return Iterators.any(
        Iterators.map(target -> subsumes(target, vn), context.target_varnames)
    )
end

function is_target_varname(context::GibbsContext, vns::AbstractArray{<:VarName})
    return all(Base.Fix1(is_target_varname, context), vns)
end

# Tilde pipeline
function DynamicPPL.tilde_assume(context::GibbsContext, right, vn, vi)
    if is_target_varname(context, vn)
        # Fall back to the default behavior.
        return DynamicPPL.tilde_assume(DynamicPPL.childcontext(context), right, vn, vi)
    elseif has_conditioned_gibbs(context, vn)
        # Short-circuits the tilde assume if `vn` is present in `context`.
        value = get_conditioned_gibbs(context, vn)
        # TODO(mhauru) Is the call to logpdf correct if context.context is not
        # DefaultContext?
        return value, logpdf(right, value), vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            vn,
            context.global_varinfo[],
        )
        context.global_varinfo[] = new_global_vi
        return value, lp, vi
    end
end

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, vn, vi
)
    if is_target_varname(context, vn)
        # Fall back to the default behavior.
        return DynamicPPL.tilde_assume(
            rng, DynamicPPL.childcontext(context), sampler, right, vn, vi
        )
    elseif has_conditioned_gibbs(context, vn)
        # Short-circuits the tilde assume if `vn` is present in `context`.
        value = get_conditioned_gibbs(context, vn)
        # TODO(mhauru) Is the call to logpdf correct if context.context is not
        # DefaultContext?
        return value, logpdf(right, value), vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            rng,
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            vn,
            context.global_varinfo[],
        )
        context.global_varinfo[] = new_global_vi
        return value, lp, vi
    end
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
    if is_target_varname(context, vns)
        # Fall back to the default behavior.
        return DynamicPPL.dot_tilde_assume(
            DynamicPPL.childcontext(context), right, left, vns, vi
        )
    elseif has_conditioned_gibbs(context, vns)
        # Short-circuit the tilde assume if `vn` is present in `context`.
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        return value, broadcast_logpdf(right, value), vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            left,
            vns,
            context.global_varinfo[],
        )
        context.global_varinfo[] = new_global_vi
        return value, lp, vi
    end
end

function DynamicPPL.dot_tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, left, vns, vi
)
    if is_target_varname(context, vns)
        # Fall back to the default behavior.
        return DynamicPPL.dot_tilde_assume(
            rng, DynamicPPL.childcontext(context), sampler, right, left, vns, vi
        )
    elseif has_conditioned_gibbs(context, vns)
        # Short-circuit the tilde assume if `vn` is present in `context`.
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        return value, broadcast_logpdf(right, value), vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            rng,
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            left,
            vns,
            context.global_varinfo[],
        )
        context.global_varinfo[] = new_global_vi
        return value, lp, vi
    end
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
    make_conditional(model, target_variables, varinfo)

Return a new, conditioned model for a component of a Gibbs sampler.

# Arguments
- `model::DynamicPPL.Model`: The model to condition.
- `target_variables::AbstractVector{<:VarName}`: The target variables of the component
sampler. These will _not_ conditioned.
- `varinfo::DynamicPPL.AbstractVarInfo`: Values for all variables in the model. All the
values in `varinfo` but not in `target_variables` will be conditioned to the values they
have in `varinfo`.
"""
function make_conditional(
    model::DynamicPPL.Model, target_variables::AbstractVector{<:VarName}, varinfo
)
    gibbs_context = GibbsContext(target_variables, Ref(varinfo), model.context)
    return DynamicPPL.contextualize(model, gibbs_context), gibbs_context
end

# HACK: Allows us to support either passing in an implementation of `AbstractMCMC.AbstractSampler`
# or an `AbstractInferenceAlgorithm`.
wrap_algorithm_maybe(x) = x
wrap_algorithm_maybe(x::InferenceAlgorithm) = DynamicPPL.Sampler(x)

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
    vi = DynamicPPL.VarInfo(rng, model)
    if initial_params !== nothing
        vi = DynamicPPL.unflatten(vi, initial_params)
    end

    # Initialise each component sampler in turn, collect all their states.
    states = []
    for (varnames_local, sampler_local) in zip(varnames, samplers)
        varnames_local = _maybevec(varnames_local)
        # Get the initial values for this component sampler.
        vi_local = DynamicPPL.subset(vi, varnames_local)
        initial_params_local = initial_params === nothing ? nothing : vi_local[:]

        # Construct the conditional model.
        model_local, context_local = make_conditional(model, varnames_local, vi)

        # Take initial step.
        _, new_state_local = AbstractMCMC.step(
            rng,
            model_local,
            sampler_local;
            # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
            # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
            initial_params=initial_params_local,
            kwargs...,
        )
        new_vi_local = varinfo(new_state_local)
        # TODO(mhauru) Can we remove the invlinking?
        new_vi_local = if DynamicPPL.istrans(new_vi_local)
            DynamicPPL.invlink(new_vi_local, sampler_local, model_local)
        else
            new_vi_local
        end
        # This merges in any new variables that were introduced during the step, but that
        # were not in the domain of the current sampler.
        vi = merge(vi, context_local.global_varinfo[])
        # This merges the latest values for all the variables in the current sampler.
        vi = merge(vi, new_vi_local)
        push!(states, new_state_local)
    end
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs},
    state::GibbsState;
    kwargs...,
)
    vi = varinfo(state)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    # TODO: move this into a recursive function so we can unroll when reasonable?
    for index in 1:length(samplers)
        # Take the inner step.
        vi, new_state_local = gibbs_step_inner(
            rng, model, varnames, samplers, states, vi, index; kwargs...
        )
        states = Accessors.setindex(states, new_state_local, index)
    end
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
    return setlogp!!(state, vi_new.logp[])
end

# TODO(mhauru) Would really like to type constrain this to something like AbstractMCMCState
# if such a thing existed.
function DynamicPPL.setlogp!!(state, logp)
    try
        new_vi = setlogp!!(state.vi, logp)
        if new_vi !== state.vi
            return Accessors.set(state, Accessors.PropertyLens{:vi}(), new_vi)
        else
            return state
        end
    catch
        error(
            "Unable to set `state.vi` for a $(typeof(state)). " *
            "Consider writing a method for `setlogp!!` for this type.",
        )
    end
end

function DynamicPPL.setlogp!!(state::TuringState, logp)
    return TuringState(setlogp!!(state.state, logp), logp)
end

# TODO(mhauru) In the general case, which arguments are really needed for reset_state!!?
# The current list is a guess, but I think some might be unnecessary.
"""
    reset_state!!(rng, model, sampler, state, varinfo, sampler_previous, state_previous)

Return an updated state for a component sampler.

This takes into account changes caused by other Gibbs components. The default implementation
is to try to set the `vi` field of `state` to `varinfo`. If this is not the right thing to
do, a method should be implemented for the specific type of `state`.

# Arguments
- `model::DynamicPPL.Model`: The model as seen by this component sampler. Variables not
sampled by this component sampler have been conditioned with a `GibbsContext`.
- `sampler::DynamicPPL.Sampler`: The current component sampler.
- `state`: The state of this component sampler from its previous iteration.
- `varinfo::DynamicPPL.AbstractVarInfo`: The current `VarInfo`, subsetted to the variables
sampled by this component sampler.
- `sampler_previous::DynamicPPL.Sampler`: The previous sampler in the Gibbs chain.
- `state_previous`: The state returned by the previous sampler.

# Returns
An updated state of the same type as `state`. It should have variables set to the values in
`varinfo`, and any other relevant updates done.
"""
function reset_state!!(
    model, sampler, state, varinfo::AbstractVarInfo, sampler_previous, state_previous
)
    # In the fallback implementation we guess that `state` has a field called `vi` we can
    # set. Fingers crossed!
    try
        return Accessors.set(state, Accessors.PropertyLens{:vi}(), varinfo)
    catch
        error(
            "Unable to set `state.vi` for a $(typeof(state)). " *
            "Consider writing a method for reset_state!! for this type.",
        )
    end
end

# Some samplers use a VarInfo directly as the state. In that case, there's little to do in
# `reset_state!!`.
function reset_state!!(
    model,
    sampler,
    state::AbstractVarInfo,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    return varinfo
end

function reset_state!!(
    model,
    sampler,
    state::TuringState,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    new_inner_state = reset_state!!(
        model, sampler, state.state, varinfo, sampler_previous, state_previous
    )
    return TuringState(new_inner_state, state.logdensity)
end

function reset_state!!(
    model,
    sampler,
    state::HMCState,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    θ_new = varinfo[:]
    hamiltonian = get_hamiltonian(model, sampler, varinfo, state, length(θ_new))

    # Update the parameter values in `state.z`.
    # TODO: Avoid mutation
    z = state.z
    resize!(z.θ, length(θ_new))
    z.θ .= θ_new
    return HMCState(varinfo, state.i, state.kernel, hamiltonian, z, state.adaptor)
end

function reset_state!!(
    model,
    sampler,
    state::AdvancedHMC.HMCState,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    hamiltonian = AdvancedHMC.Hamiltonian(
        state.metric, DynamicPPL.LogDensityFunction(model)
    )
    θ_new = varinfo[:]
    # Set the momentum to zero, since we have no idea what it should be at the new parameter
    # values.
    return Accessors.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, θ_new, zero(θ_new)
    )
end

function reset_state!!(
    model,
    sampler,
    state::AdvancedMH.Transition,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    # TODO(mhauru) Setting the last argument like this seems a bit suspect, since the
    # current values for the parameters might not have come from this sampler at all.
    # I don't see a better way though.
    return AdvancedMH.Transition(varinfo[:], varinfo.logp[], state.accepted)
end

function reset_state!!(
    model,
    sampler,
    state::PGState,
    varinfo::AbstractVarInfo,
    sampler_previous,
    state_previous,
)
    return PGState(varinfo, state.rng)
end

function gibbs_step_inner(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varnames,
    samplers,
    states,
    vi,
    index;
    kwargs...,
)
    sampler_local = samplers[index]
    state_local = states[index]
    varnames_local = _maybevec(varnames[index])

    # TODO(mhauru) Can we remove the invlinking?
    vi = DynamicPPL.istrans(vi) ? DynamicPPL.invlink(vi, model) : vi

    # Construct the conditional model and the varinfo that this sampler should use.
    model_local, context_local = make_conditional(model, varnames_local, vi)
    varinfo_local = subset(vi, varnames_local)
    # TODO(mhauru) Can we remove the below, if get rid of all the invlinking?
    # If the varinfo of the previous state from this sampler is linked, we should link the
    # new varinfo too.
    if DynamicPPL.istrans(varinfo(state_local))
        varinfo_local = DynamicPPL.link(varinfo_local, sampler_local, model_local)
    end

    # Extract the previous sampler and state.
    sampler_previous = samplers[index == 1 ? length(samplers) : index - 1]
    state_previous = states[index == 1 ? length(states) : index - 1]

    # Set the state of the current sampler, accounting for any changes made by other
    # samplers.
    state_local = reset_state!!(
        model_local,
        sampler_local,
        state_local,
        varinfo_local,
        sampler_previous,
        state_previous,
    )
    if gibbs_requires_recompute_logprob(
        model_local, sampler_local, sampler_previous, state_local, state_previous
    )
        state_local = recompute_logprob!!(rng, model_local, sampler_local, state_local)
    end

    # Take a step with the local sampler.
    new_state_local = last(
        AbstractMCMC.step(rng, model_local, sampler_local, state_local; kwargs...)
    )

    new_vi_local = varinfo(new_state_local)
    # This merges in any new variables that were introduced during the step, but that
    # were not in the domain of the current sampler.
    new_vi = merge(vi, context_local.global_varinfo[])
    # This merges the latest values for all the variables in the current sampler.
    new_vi = merge(new_vi, new_vi_local)
    new_vi = setlogp!!(new_vi, new_vi_local.logp[])
    return new_vi, new_state_local
end
