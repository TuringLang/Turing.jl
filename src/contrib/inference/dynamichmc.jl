###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

"""
    DynamicNUTSState

State of the [`DynamicNUTS`](@ref) sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DynamicNUTSState{L,C,M,S}
    logdensity::L
    "Cache of sample, log density, and gradient of log density evaluation."
    cache::C
    metric::M
    stepsize::S
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::LogDensityModel,
    spl::DynamicNUTS;
    init_params = nothing,
    kwargs...
)
    # Unpack model
    ℓ = model.logdensity

    # Perform initial step.
    results = mcmc_keep_warmup(
        rng,
        ℓ,
        0;
        initialization = (q = init_params,),
        reporter = NoProgressReport(),
    )
    steps = mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    transition, _ = mcmc_next_step(steps, results.final_warmup_state.Q)
    state = DynamicNUTSState(ℓ, transition, steps.H.κ, steps.ϵ)

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::LogDensityModel,
    spl::DynamicHMC.NUTS,
    state::DynamicNUTSState;
    kwargs...
)
    # Compute next sample.
    ℓ = state.logdensity
    steps = DynamicHMC.mcmc_steps(
        rng,
        spl,
        state.metric,
        ℓ,
        state.stepsize,
    )
    transition, _ = DynamicHMC.mcmc_next_step(steps, state.cache)
    newstate = DynamicNUTSState(ℓ, transition, state.metric, state.stepsize)

    return transition, newstate
end
