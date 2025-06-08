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
struct DynamicNUTSState{V<:DynamicPPL.AbstractVarInfo,C,M,S}
    vi::V
    "Cache of sample, log density, and gradient of log density evaluation."
    cache::C
    metric::M
    stepsize::S
end

function DynamicPPL.initialsampler(::DynamicPPL.Sampler{<:DynamicNUTS})
    return DynamicPPL.SampleFromUniform()
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    ldf::DynamicPPL.LogDensityFunction,
    spl::DynamicPPL.Sampler{<:DynamicNUTS};
    kwargs...,
)
    vi = ldf.varinfo

    # Perform initial step.
    results = DynamicHMC.mcmc_keep_warmup(
        rng, ldf, 0; initialization=(q=vi[:],), reporter=DynamicHMC.NoProgressReport()
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Update the variables.
    vi = DynamicPPL.unflatten(vi, Q.q)
    vi = DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create first sample and state.
    sample = Turing.Inference.Transition(ldf.model, vi)
    state = DynamicNUTSState(vi, Q, steps.H.κ, steps.ϵ)

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    ldf::DynamicPPL.LogDensityFunction,
    spl::DynamicPPL.Sampler{<:DynamicNUTS},
    state::DynamicNUTSState;
    kwargs...,
)
    # Compute next sample.
    vi = state.vi
    steps = DynamicHMC.mcmc_steps(rng, spl.alg.sampler, state.metric, ldf, state.stepsize)
    Q, _ = DynamicHMC.mcmc_next_step(steps, state.cache)

    # Update the variables.
    vi = DynamicPPL.unflatten(vi, Q.q)
    vi = DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create next sample and state.
    sample = Turing.Inference.Transition(ldf.model, vi)
    newstate = DynamicNUTSState(vi, Q, state.metric, state.stepsize)

    return sample, newstate
end

end
