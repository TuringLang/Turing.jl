module DynamicHMCExt
###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###


if isdefined(Base, :get_extension)
    import DynamicHMC
    using Turing
    using Turing: AbstractMCMC, Random, LogDensityProblems, DocStringExtensions
    using Turing.Inference: LogDensityProblemsAD
else
    import ..DynamicHMC
    using ..Turing
    using ..Turing: AbstractMCMC, Random, LogDensityProblems 
    using ..Turing.Inference: LogDensityProblemsAD
end

# Wraps DynamicHMC as an AbstractSampler
struct DynamicNUTS{S<:DynamicHMC.NUTS} <: AbstractMCMC.AbstractSampler
    sampler::S
end

DynamicNUTS() = DynamicNUTS(DynamicHMC.NUTS())
Turing.externalsampler(spl::DynamicHMC.NUTS) = Turing.externalsampler(DynamicNUTS(spl))

"""
    DynamicNUTSState

State of the [`DynamicNUTS`](@ref) sampler.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DynamicNUTSState{L,C,M,S}
    logdensity::L
    "Cache of sample, log density, and gradient of log density evaluation."
    cache::C
    metric::M
    stepsize::S
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::DynamicNUTS;
    init_params = nothing,
    kwargs...
)
    # Unpack model
    # We wrap it again in ADgradient 
    ℓ = LogDensityProblemsAD.ADgradient(model.logdensity)

    # Make init params if nothing
    if init_params === nothing
        d = LogDensityProblems.dimension(model.logdensity)
        init_params = randn(rng, d)
    end 

    # Perform initial step.
    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        ℓ,
        0;
        initialization = (q = init_params,),
        reporter = DynamicHMC.NoProgressReport(),
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    transition, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)
    state = DynamicNUTSState(ℓ, transition, steps.H.κ, steps.ϵ)
    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::DynamicNUTS,
    state::DynamicNUTSState;
    kwargs...
)
    # Compute next sample.
    ℓ = state.logdensity
    steps = DynamicHMC.mcmc_steps(
        rng,
        spl.sampler,
        state.metric,
        ℓ,
        state.stepsize,
    )
    transition, _ = DynamicHMC.mcmc_next_step(steps, state.cache)
    newstate = DynamicNUTSState(ℓ, transition, state.metric, state.stepsize)

    return transition, newstate
end

Turing.Inference.getparams(transition::DynamicHMC.EvaluatedLogDensity) = transition.q

end # module