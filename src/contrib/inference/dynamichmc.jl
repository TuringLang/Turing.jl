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

struct DynamicHMCLogDensity{M<:Model,S<:Sampler{<:DynamicNUTS},V<:AbstractVarInfo}
    model::M
    sampler::S
    varinfo::V
end

function DynamicHMC.dimension(ℓ::DynamicHMCLogDensity)
    return length(ℓ.varinfo[ℓ.sampler])
end

function DynamicHMC.capabilities(::Type{<:DynamicHMCLogDensity})
    return DynamicHMC.LogDensityOrder{1}()
end

function DynamicHMC.logdensity_and_gradient(
    ℓ::DynamicHMCLogDensity,
    x::AbstractVector,
)
    return gradient_logp(x, ℓ.varinfo, ℓ.model, ℓ.sampler)
end

"""
    DynamicNUTSState

State of the [`DynamicNUTS`](@ref) sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DynamicNUTSState{V<:AbstractVarInfo,C,M,S}
    vi::V
    "Cache of sample, log density, and gradient of log density."
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
    # Update the previous evaluation.
    ℓ = DynamicHMCLogDensity(model, spl, varinfo)
    Q = DynamicHMC.evaluate_ℓ(ℓ, varinfo[spl])
    return DynamicNUTSState(varinfo, Q, state.metric, state.stepsize)
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
        DynamicPPL.link!(vi, spl)
        model(rng, vi, spl)
    end

    # Perform initial step.
    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        DynamicHMCLogDensity(model, spl, vi),
        0;
        initialization = (q = vi[spl],),
        reporter = DynamicHMC.NoProgressReport(),
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Update the variables.
    vi[spl] = Q.q
    DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create first sample and state.
    sample = Transition(vi)
    state = DynamicNUTSState(vi, Q, steps.H.κ, steps.ϵ)

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
    ℓ = DynamicHMCLogDensity(model, spl, vi)
    steps = DynamicHMC.mcmc_steps(
        rng,
        DynamicHMC.NUTS(),
        state.metric,
        ℓ,
        state.stepsize,
    )
    Q, _ = DynamicHMC.mcmc_next_step(steps, state.cache)

    # Update the variables.
    vi[spl] = Q.q
    DynamicPPL.setlogp!!(vi, Q.ℓq)

    # Create next sample and state.
    sample = Transition(vi)
    newstate = DynamicNUTSState(vi, Q, state.metric, state.stepsize)

    return sample, newstate
end
