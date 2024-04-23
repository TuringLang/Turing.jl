struct TuringState{S,F}
    state::S
    logdensity::F
end

state_to_turing(f::DynamicPPL.LogDensityFunction, state) = TuringState(state, f)
function transition_to_turing(f::DynamicPPL.LogDensityFunction, transition)
    # TODO: We should probably rename this `getparams` since it returns something
    # very different from `Turing.Inference.getparams`.
    θ = getparams(f.model, transition)
    varinfo = DynamicPPL.unflatten(f.varinfo, θ)
    return Transition(f.model, varinfo, transition)
end

function varinfo(state::TuringState)
    θ = getparams(state.logdensity.model, state.state)
    # TODO: Do we need to link here first?
    return DynamicPPL.unflatten(state.logdensity.varinfo, θ)
end

# NOTE: Only thing that depends on the underlying sampler.
# Something similar should be part of AbstractMCMC at some point:
# https://github.com/TuringLang/AbstractMCMC.jl/pull/86
getparams(::DynamicPPL.Model, transition::AdvancedHMC.Transition) = transition.z.θ
function getparams(model::DynamicPPL.Model, state::AdvancedHMC.HMCState)
    return getparams(model, state.transition)
end
getstats(transition::AdvancedHMC.Transition) = transition.stat

getparams(::DynamicPPL.Model, transition::AdvancedMH.Transition) = transition.params

getvarinfo(f::DynamicPPL.LogDensityFunction) = f.varinfo
getvarinfo(f::LogDensityProblemsAD.ADGradientWrapper) = getvarinfo(parent(f))

setvarinfo(f::DynamicPPL.LogDensityFunction, varinfo) = Setfield.@set f.varinfo = varinfo
setvarinfo(f::LogDensityProblemsAD.ADGradientWrapper, varinfo) = setvarinfo(parent(f), varinfo)

"""
    recompute_logprob!!(rng, model, sampler, state)

Recompute the log-probability of the `model` based on the given `state` and return the resulting state.
"""
function recompute_logprob!!(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:ExternalSampler},
    state
)
    # Re-using the log-density function from the `state` and updating only the `model` field.
    f = state.logdensity
    f = Setfield.@set f.model = model
    # Recompute the log-probability with the new `model`.
    state_inner = recompute_logprob!!(
        rng,
        AbstractMCMC.LogDensityModel(f),
        sampler.alg.sampler,
        state.state
    )
    return state_to_turing(f, state_inner)
end

function recompute_logprob!!(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AdvancedHMC.AbstractHMCSampler,
    state::AdvancedHMC.HMCState
)
    # Construct hamiltionian.
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, model)
    # Re-compute the log-probability and gradient.
    return Setfield.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian,
        state.transition.z.θ,
        state.transition.z.r,
    )
end

# TODO: Do we also support `resume`, etc?
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler};
    kwargs...
)
    sampler = sampler_wrapper.alg.sampler

    # Create a log-density function with an implementation of the
    # gradient so we ensure that we're using the same AD backend as in Turing.
    f = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(model))

    # Link the varinfo.
    f = setvarinfo(f, DynamicPPL.link!!(getvarinfo(f), model))

    # Then just call `AdvancedHMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler; kwargs...
    )

    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler},
    state::TuringState;
    kwargs...
)
    sampler = sampler_wrapper.alg.sampler
    f = state.logdensity

    # Then just call `AdvancedHMC.step` with the right arguments.
    transition_inner, state_inner = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(f), sampler, state.state; kwargs...
    )

    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end
