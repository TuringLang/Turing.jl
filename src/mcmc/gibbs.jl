"""
    isgibbscomponent(alg::Union{InferenceAlgorithm, AbstractMCMC.AbstractSampler})

Return a boolean indicating whether `alg` is a valid component for a Gibbs sampler.

Defaults to `false` if no method has been defined for a particular algorithm type.
"""
isgibbscomponent(::InferenceAlgorithm) = false
isgibbscomponent(spl::ExternalSampler) = isgibbscomponent(spl.sampler)
isgibbscomponent(spl::Sampler) = isgibbscomponent(spl.alg)

isgibbscomponent(::ESS) = true
isgibbscomponent(::HMC) = true
isgibbscomponent(::HMCDA) = true
isgibbscomponent(::NUTS) = true
isgibbscomponent(::MH) = true
isgibbscomponent(::PG) = true
isgibbscomponent(::AdvancedHMC.HMC) = true
isgibbscomponent(::AdvancedMH.MetropolisHastings) = true

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
"""
    GibbsContext(target_varnames, global_varinfo, context)

A context used in the implementation of the Turing.jl Gibbs sampler.

There will be one `GibbsContext` for each iteration of a component sampler.

# Fields
$(FIELDS)
"""
struct GibbsContext{VNs,GVI<:Ref{<:AbstractVarInfo},Ctx<:DynamicPPL.AbstractContext} <:
       DynamicPPL.AbstractContext
    """
    a collection of `VarName`s that the current component sampler is sampling.
    For them, `GibbsContext` will just pass tilde_assume calls to its child context.
    For other variables, their values will be fixed to the values they have in
    `global_varinfo`.
    """
    target_varnames::VNs
    """
    a `Ref` to the global `AbstractVarInfo` object that holds values for all variables, both
    those fixed and those being sampled. We use a `Ref` because this field may need to be
    updated if new variables are introduced.
    """
    global_varinfo::GVI
    """
    the child context that tilde calls will eventually be passed onto.
    """
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

get_global_varinfo(context::GibbsContext) = context.global_varinfo[]

function set_global_varinfo!(context::GibbsContext, new_global_varinfo)
    context.global_varinfo[] = new_global_varinfo
    return nothing
end

# has and get
function has_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.haskey(get_global_varinfo(context), vn)
end
function has_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    num_conditioned = count(Iterators.map(Base.Fix1(has_conditioned_gibbs, context), vns))
    if (num_conditioned != 0) && (num_conditioned != length(vns))
        error(
            "Some but not all of the variables in `vns` have been conditioned on. " *
            "Having mixed conditioning like this is not supported in GibbsContext.",
        )
    end
    return num_conditioned > 0
end

function get_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return get_global_varinfo(context)[vn]
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
    num_target = count(Iterators.map(Base.Fix1(is_target_varname, context), vns))
    if (num_target != 0) && (num_target != length(vns))
        error(
            "Some but not all of the variables in `vns` are target variables. " *
            "Having mixed targeting like this is not supported in GibbsContext.",
        )
    end
    return num_target > 0
end

# Tilde pipeline
function DynamicPPL.tilde_assume(context::GibbsContext, right, vn, vi)
    return if is_target_varname(context, vn)
        # Fall back to the default behavior.
        DynamicPPL.tilde_assume(DynamicPPL.childcontext(context), right, vn, vi)
    elseif has_conditioned_gibbs(context, vn)
        # Short-circuit the tilde assume if `vn` is present in `context`.
        value = get_conditioned_gibbs(context, vn)
        # TODO(mhauru) Is the call to logpdf correct if context.context is not
        # DefaultContext?
        value, logpdf(right, value), vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            DynamicPPL.childcontext(context),
            DynamicPPL.SampleFromPrior(),
            right,
            vn,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

# As above but with an RNG.
function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, vn, vi
)
    # See comment in the above, rng-less version of this method for an explanation.
    return if is_target_varname(context, vn)
        DynamicPPL.tilde_assume(
            rng, DynamicPPL.childcontext(context), sampler, right, vn, vi
        )
    elseif has_conditioned_gibbs(context, vn)
        value = get_conditioned_gibbs(context, vn)
        # TODO(mhauru) As above, is logpdf correct if context.context is not DefaultContext?
        value, logpdf(right, value), vi
    else
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            rng,
            DynamicPPL.childcontext(context),
            DynamicPPL.SampleFromPrior(),
            right,
            vn,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
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

# Like the above tilde_assume methods, but with dot_tilde_assume and broadcasting of logpdf.
# See comments there for more details.
function DynamicPPL.dot_tilde_assume(context::GibbsContext, right, left, vns, vi)
    return if is_target_varname(context, vns)
        DynamicPPL.dot_tilde_assume(DynamicPPL.childcontext(context), right, left, vns, vi)
    elseif has_conditioned_gibbs(context, vns)
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        # TODO(mhauru) As above, is logpdf correct if context.context is not DefaultContext?
        value, broadcast_logpdf(right, value), vi
    else
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            left,
            vns,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

# As above but with an RNG.
function DynamicPPL.dot_tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, left, vns, vi
)
    return if is_target_varname(context, vns)
        DynamicPPL.dot_tilde_assume(
            rng, DynamicPPL.childcontext(context), sampler, right, left, vns, vi
        )
    elseif has_conditioned_gibbs(context, vns)
        value = reconstruct_getvalue(right, get_conditioned_gibbs(context, vns))
        # TODO(mhauru) As above, is logpdf correct if context.context is not DefaultContext?
        value, broadcast_logpdf(right, value), vi
    else
        prior_sampler = DynamicPPL.SampleFromPrior()
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            rng,
            DynamicPPL.childcontext(context),
            prior_sampler,
            right,
            left,
            vns,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

"""
    make_conditional(model, target_variables, varinfo)

Return a new, conditioned model for a component of a Gibbs sampler.

# Arguments
- `model::DynamicPPL.Model`: The model to condition.
- `target_variables::AbstractVector{<:VarName}`: The target variables of the component
sampler. These will _not_ be conditioned.
- `varinfo::DynamicPPL.AbstractVarInfo`: Values for all variables in the model. All the
values in `varinfo` but not in `target_variables` will be conditioned to the values they
have in `varinfo`.

# Returns
- A new model with the variables _not_ in `target_variables` conditioned.
- The `GibbsContext` object that will be used to condition the variables. This is necessary
because evaluation can mutate its `global_varinfo` field, which we need to access later.
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

    function Gibbs(varnames, samplers)
        if length(varnames) != length(samplers)
            throw(ArgumentError("Number of varnames and samplers must match."))
        end
        for spl in samplers
            if !isgibbscomponent(spl)
                msg = "All samplers must be valid Gibbs components, $(spl) is not."
                throw(ArgumentError(msg))
            end
        end
        return new{typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
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

varinfo(state::GibbsState) = state.vi

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

    # Run the model once to get the varnames present + initial values to condition on.
    vi = DynamicPPL.VarInfo(rng, model)
    if initial_params !== nothing
        vi = DynamicPPL.unflatten(vi, initial_params)
    end

    # Initialise each component sampler in turn, collect all their states.
    states = []
    for (varnames_local, sampler_local) in zip(varnames, samplers)
        varnames_local = _maybevec(varnames_local)
        # Get the initial values for this component sampler.
        initial_params_local = if initial_params === nothing
            nothing
        else
            DynamicPPL.subset(vi, varnames_local)[:]
        end

        # Construct the conditioned model.
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
        # TODO(mhauru) Remove the below loop once samplers no longer depend on selectors.
        # For some reason not having this in place was causing trouble for ESS, but not for
        # other samplers. I didn't get to the bottom of it.
        for vn in keys(new_vi_local)
            DynamicPPL.setgid!(new_vi_local, sampler_local.selector, vn)
        end
        # Merge in any new variables that were introduced during the step, but that
        # were not in the domain of the current sampler.
        vi = merge(vi, context_local.global_varinfo[])
        # Merge the new values for all the variables sampled by the current sampler.
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
        sampler_local = samplers[index]
        state_local = states[index]
        varnames_local = _maybevec(varnames[index])
        vi, new_state_local = gibbs_step_inner(
            rng, model, varnames_local, sampler_local, state_local, vi; kwargs...
        )
        states = Accessors.setindex(states, new_state_local, index)
    end
    return Transition(model, vi), GibbsState(vi, states)
end

"""
    setparams_varinfo!!(model, sampler::Sampler, state, params::AbstractVarInfo)

A lot like AbstractMCMC.setparams!!, but instead of taking a vector of parameters, takes an
`AbstractVarInfo` object. Also takes the `sampler` as an argument. By default, falls back to
`AbstractMCMC.setparams!!(model, state, params[:])`.

`model` is typically a `DynamicPPL.Model`, but can also be e.g. an
`AbstractMCMC.LogDensityModel`.
"""
function setparams_varinfo!!(model, ::Sampler, state, params::AbstractVarInfo)
    return AbstractMCMC.setparams!!(model, state, params[:])
end

function setparams_varinfo!!(
    model::DynamicPPL.Model, sampler::Sampler{<:MH}, state::VarInfo, params::AbstractVarInfo
)
    # The state is already a VarInfo, so we can just return `params`, but first we need to
    # update its logprob.
    # NOTE: Using `leafcontext(model.context)` here is a no-op, as it will be concatenated
    # with `model.context` before hitting `model.f`.
    return last(DynamicPPL.evaluate!!(model, params, DynamicPPL.leafcontext(model.context)))
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:ESS},
    state::VarInfo,
    params::AbstractVarInfo,
)
    # The state is already a VarInfo, so we can just return `params`, but first we need to
    # update its logprob.
    # Note the use of LikelihoodContext, regardless of what context `model` has. This is
    # specific to ESS as a sampler.
    return last(DynamicPPL.evaluate!!(model, params, DynamicPPL.LikelihoodContext()))
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:ExternalSampler},
    state::TuringState,
    params::AbstractVarInfo,
)
    logdensity = DynamicPPL.setmodel(state.logdensity, model, sampler.alg.adtype)
    new_inner_state = setparams_varinfo!!(
        AbstractMCMC.LogDensityModel(logdensity), sampler, state.state, params
    )
    return TuringState(new_inner_state, logdensity)
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:Hamiltonian},
    state::HMCState,
    params::AbstractVarInfo,
)
    θ_new = params[:]
    hamiltonian = get_hamiltonian(model, sampler, params, state, length(θ_new))

    # Update the parameter values in `state.z`.
    # TODO: Avoid mutation
    z = state.z
    resize!(z.θ, length(θ_new))
    z.θ .= θ_new
    return HMCState(params, state.i, state.kernel, hamiltonian, z, state.adaptor)
end

function setparams_varinfo!!(
    model::DynamicPPL.Model, sampler::Sampler{<:PG}, state::PGState, params::AbstractVarInfo
)
    return PGState(params, state.rng)
end

function gibbs_step_inner(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varnames_local,
    sampler_local,
    state_local,
    global_vi;
    kwargs...,
)
    # Construct the conditional model and the varinfo that this sampler should use.
    model_local, context_local = make_conditional(model, varnames_local, global_vi)
    varinfo_local = subset(global_vi, varnames_local)
    # TODO(mhauru) Remove the below loop once samplers no longer depend on selectors.
    # For some reason not having this in place was causing trouble for ESS, but not for
    # other samplers. I didn't get to the bottom of it.
    for vn in keys(varinfo_local)
        DynamicPPL.setgid!(varinfo_local, sampler_local.selector, vn)
    end

    # TODO(mhauru) The below may be overkill. If the varnames for this sampler are not
    # sampled by other samplers, we don't need to `setparams`, but could rather simply
    # recompute the log probability. More over, in some cases the recomputation could also
    # be avoided, if e.g. the previous sampler has done all the necessary work already.
    # However, we've judged that doing any caching or other tricks to avoid this now would
    # be premature optimization. In most use cases of Gibbs a single model call here is not
    # going to be a significant expense anyway.
    # Set the state of the current sampler, accounting for any changes made by other
    # samplers.
    state_local = setparams_varinfo!!(
        model_local, sampler_local, state_local, varinfo_local
    )

    # Take a step with the local sampler.
    new_state_local = last(
        AbstractMCMC.step(rng, model_local, sampler_local, state_local; kwargs...)
    )

    new_vi_local = varinfo(new_state_local)
    # Merge the latest values for all the variables in the current sampler.
    new_global_vi = merge(get_global_varinfo(context_local), new_vi_local)
    new_global_vi = setlogp!!(new_global_vi, getlogp(new_vi_local))
    return new_global_vi, new_state_local
end
