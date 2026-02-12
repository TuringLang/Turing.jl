module TuringDynamicHMCExt
###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

using DynamicHMC: DynamicHMC
using Turing
using Turing: AbstractMCMC, Random, LogDensityProblems, DynamicPPL
using Turing.Inference: ADTypes, TYPEDFIELDS

"""
    DynamicNUTS

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.

To use it, make sure you have DynamicHMC package (version >= 2) loaded:
```julia
using DynamicHMC
```
"""
struct DynamicNUTS{AD,T<:DynamicHMC.NUTS} <: Turing.Inference.Hamiltonian
    sampler::T
    adtype::AD
end

DynamicNUTS() = DynamicNUTS(DynamicHMC.NUTS())
DynamicNUTS(spl) = DynamicNUTS(spl, Turing.DEFAULT_ADTYPE)
Turing.externalsampler(spl::DynamicHMC.NUTS) = DynamicNUTS(spl)

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
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicNUTS;
    initial_params,
    kwargs...,
)
    # Define log-density function.
    # TODO(penelopeysm) We need to check that the initial parameters are valid. Same as how
    # we do it for HMC
    _, vi = DynamicPPL.init!!(
        rng, model, DynamicPPL.VarInfo(), initial_params, DynamicPPL.LinkAll()
    )
    ℓ = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.adtype
    )

    # Perform initial step.
    results = DynamicHMC.mcmc_keep_warmup(
        rng, ℓ, 0; initialization=(q=vi[:],), reporter=DynamicHMC.NoProgressReport()
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Create first sample and state.
    sample = DynamicPPL.ParamsWithStats(Q.q, ℓ)
    state = DynamicNUTSState(ℓ, Q, steps.H.κ, steps.ϵ)

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicNUTS,
    state::DynamicNUTSState;
    kwargs...,
)
    # Compute next sample.
    ℓ = state.logdensity
    steps = DynamicHMC.mcmc_steps(rng, spl.sampler, state.metric, ℓ, state.stepsize)
    Q, _ = DynamicHMC.mcmc_next_step(steps, state.cache)

    # Create next sample and state.
    sample = DynamicPPL.ParamsWithStats(Q.q, ℓ)
    newstate = DynamicNUTSState(ℓ, Q, state.metric, state.stepsize)

    return sample, newstate
end

end
