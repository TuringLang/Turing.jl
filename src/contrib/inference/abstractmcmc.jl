struct TuringState{S,F}
    state::S
    logdensity::F
end

state_to_turing(f::DynamicPPL.LogDensityFunction, state) = TuringState(state, f)
function transition_to_turing(f::DynamicPPL.LogDensityFunction, transition)
    θ = getparams(transition)
    varinfo = DynamicPPL.unflatten(f.varinfo, θ)
    # TODO: `deepcopy` is overkill; make more efficient.
    varinfo = DynamicPPL.invlink!!(deepcopy(varinfo), f.model)
    return Transition(varinfo, transition)
end

# NOTE: Only thing that depends on the underlying sampler.
# Something similar should be part of AbstractMCMC at some point:
# https://github.com/TuringLang/AbstractMCMC.jl/pull/86
getparams(transition::AdvancedHMC.Transition) = transition.z.θ
getstats(transition::AdvancedHMC.Transition) = transition.stat

getparams(transition::AdvancedMH.Transition) = transition.params

getvarinfo(f::DynamicPPL.LogDensityFunction) = f.varinfo
getvarinfo(f::LogDensityProblemsAD.ADGradientWrapper) = getvarinfo(parent(f))

setvarinfo(f::DynamicPPL.LogDensityFunction, varinfo) = Setfield.@set f.varinfo = varinfo
setvarinfo(f::LogDensityProblemsAD.ADGradientWrapper, varinfo) = setvarinfo(parent(f), varinfo)

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
