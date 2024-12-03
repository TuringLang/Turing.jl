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

state_to_turing(f::LogDensityProblemsAD.ADGradientWrapper, state) = TuringState(state, f)
function transition_to_turing(f::LogDensityProblemsAD.ADGradientWrapper, transition)
    return transition_to_turing(parent(f), transition)
end

function varinfo_from_logdensityfn(f::LogDensityProblemsAD.ADGradientWrapper)
    return varinfo_from_logdensityfn(parent(f))
end
varinfo_from_logdensityfn(f::DynamicPPL.LogDensityFunction) = f.varinfo

function varinfo(state::TuringState)
    θ = getparams(DynamicPPL.getmodel(state.logdensity), state.state)
    # TODO: Do we need to link here first?
    return DynamicPPL.unflatten(varinfo_from_logdensityfn(state.logdensity), θ)
end
varinfo(state::AbstractVarInfo) = state

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
function getvarinfo(f::LogDensityProblemsAD.ADGradientWrapper)
    return getvarinfo(LogDensityProblemsAD.parent(f))
end

setvarinfo(f::DynamicPPL.LogDensityFunction, varinfo) = Accessors.@set f.varinfo = varinfo
function setvarinfo(
    f::LogDensityProblemsAD.ADGradientWrapper, varinfo, adtype::ADTypes.AbstractADType
)
    return LogDensityProblemsAD.ADgradient(
        adtype, setvarinfo(LogDensityProblemsAD.parent(f), varinfo)
    )
end

# TODO: Do we also support `resume`, etc?
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler};
    initial_state=nothing,
    initial_params=nothing,
    kwargs...,
)
    alg = sampler_wrapper.alg
    sampler = alg.sampler

    # Create a log-density function with an implementation of the
    # gradient so we ensure that we're using the same AD backend as in Turing.
    f = LogDensityProblemsAD.ADgradient(alg.adtype, DynamicPPL.LogDensityFunction(model))

    # Link the varinfo if needed.
    varinfo = getvarinfo(f)
    if requires_unconstrained_space(alg)
        if initial_params !== nothing
            # If we have initial parameters, we need to set the varinfo before linking.
            varinfo = DynamicPPL.link(DynamicPPL.unflatten(varinfo, initial_params), model)
            # Extract initial parameters in unconstrained space.
            initial_params = varinfo[:]
        else
            varinfo = DynamicPPL.link(varinfo, model)
        end
    end
    f = setvarinfo(f, varinfo, alg.adtype)

    # Then just call `AdvancedHMC.step` with the right arguments.
    if initial_state === nothing
        transition_inner, state_inner = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(f), sampler; initial_params, kwargs...
        )
    else
        transition_inner, state_inner = AbstractMCMC.step(
            rng,
            AbstractMCMC.LogDensityModel(f),
            sampler,
            initial_state;
            initial_params,
            kwargs...,
        )
    end
    # Update the `state`
    return transition_to_turing(f, transition_inner), state_to_turing(f, state_inner)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapper::Sampler{<:ExternalSampler},
    state::TuringState;
    kwargs...,
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
