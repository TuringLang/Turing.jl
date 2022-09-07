###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

"""
    DynamicNUTS

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.

To use it, make sure you have DynamicHMC package (version >= 2) loaded:
```julia
using DynamicHMC
```
""" 
struct DynamicNUTS{AD,space} <: Hamiltonian{AD} end

DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
DynamicNUTS{AD}(space::Symbol...) where AD = DynamicNUTS{AD, space}()

DynamicPPL.getspace(::DynamicNUTS{<:Any, space}) where {space} = space

"""
    DynamicNUTSState

State of the [`DynamicNUTS`](@ref) sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DynamicNUTSState{L,V<:AbstractVarInfo,C,M,S}
    logdensity::L
    vi::V
    "Cache of sample, log density, and gradient of log density evaluation."
    cache::C
    metric::M
    stepsize::S
end

# Implement interface of `Gibbs` sampler
function gibbs_state(
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    state::DynamicNUTSState,
    varinfo::AbstractVarInfo,
)
    # Update the log density function and its cached evaluation.
    ℓ = LogDensityProblems.ADgradient(Turing.LogDensityFunction(varinfo, model, spl, DynamicPPL.DefaultContext()))
    Q = DynamicHMC.evaluate_ℓ(ℓ, varinfo[spl])
    return DynamicNUTSState(ℓ, varinfo, Q, state.metric, state.stepsize)
end

DynamicPPL.initialsampler(::Sampler{<:DynamicNUTS}) = SampleFromUniform()

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    vi::AbstractVarInfo;
    kwargs...
)
    # Ensure that initial sample is in unconstrained space.
    if !DynamicPPL.islinked(vi, spl)
        vi = DynamicPPL.link!!(vi, spl, model)
        vi = last(DynamicPPL.evaluate!!(model, vi, DynamicPPL.SamplingContext(rng, spl)))
    end

    # Define log-density function.
    ℓ = LogDensityProblems.ADgradient(Turing.LogDensityFunction(vi, model, spl, DynamicPPL.DefaultContext()))

    # Perform initial step.
    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        ℓ,
        0;
        initialization = (q = vi[spl],),
        reporter = DynamicHMC.NoProgressReport(),
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Update the variables.
    vi = DynamicPPL.setindex!!(vi, Q.q, spl)
    vi = DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create first sample and state.
    sample = Transition(vi)
    state = DynamicNUTSState(ℓ, vi, Q, steps.H.κ, steps.ϵ)

    return sample, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    state::DynamicNUTSState;
    kwargs...
)
    # Compute next sample.
    vi = state.vi
    ℓ = state.logdensity
    steps = DynamicHMC.mcmc_steps(
        rng,
        DynamicHMC.NUTS(),
        state.metric,
        ℓ,
        state.stepsize,
    )
    Q, _ = DynamicHMC.mcmc_next_step(steps, state.cache)

    # Update the variables.
    vi = DynamicPPL.setindex!!(vi, Q.q, spl)
    vi = DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create next sample and state.
    sample = Transition(vi)
    newstate = DynamicNUTSState(ℓ, vi, Q, state.metric, state.stepsize)

    return sample, newstate
end
