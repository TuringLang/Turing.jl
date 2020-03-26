###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

"""
    DynamicNUTS

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package. To use it, make
sure you have the LogDensityProblems package and DynamicHMC package (version >= 2) loaded:

```julia
using LogDensityProblems, DynamicHMC
```
"""
struct DynamicNUTS{AD, space} <: Hamiltonian{AD} end

DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
DynamicNUTS{AD}(space::Symbol...) where AD = DynamicNUTS{AD, space}()

getspace(::DynamicNUTS{<:Any, space}) where {space} = space

mutable struct DynamicNUTSState{V<:VarInfo} <: AbstractSamplerState
    vi::V
end

function Sampler(
    alg::DynamicNUTS,
    model::Model,
    s::Selector=Selector()
)
    # Construct a state, using a default function.
    state = DynamicNUTSState(VarInfo(model))

    # Return a new sampler.
    return Sampler(alg, Dict{Symbol,Any}(), s, state)
end

"""
    DynamicNUTSTransition

Transition for the `DynamicNUTS` sampler.
"""
struct DynamicNUTSTransition{T,F<:AbstractFloat,QType,H,S}
    θ::T
    lp::F
    Q::QType
    hamiltonian::H
    stepsize::S
end

function additional_parameters(::Type{<:DynamicNUTSTransition})
    return [:lp]
end

# Wrapper for the log density function
struct LogDensity{M<:Model,S<:Sampler}
    model::M
    spl::S
end

function LogDensityProblems.dimension(ℓ::LogDensity)
    spl = ℓ.spl
    return length(spl.state.vi[spl])
end

function LogDensityProblems.capabilities(::Type{<:LogDensity})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity(ℓ::LogDensity, x::AbstractVector)
    sampler = ℓ.sampler
    vi = sampler.state.vi

    x_old = vi[sampler]
    lj_old = getlogp(vi)
        
    vi[sampler] = x
    runmodel!(ℓ.model, vi, sampler)
    lj = getlogp(vi)

    vi[sampler] = x_old
    setlogp!(vi, lj_old)
    
    return lj
end

function LogDensityProblems.logdensity_and_gradient(ℓ::LogDensity,
                                                    x::AbstractVector)
    spl = ℓ.spl
    return gradient_logp(x, spl.state.vi, ℓ.model, spl)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    ::Integer,
    ::Nothing;
    kwargs...
)
    # Convert to transformed space.
    vi = spl.state.vi
    if !islinked(vi, spl)
        Turing.DEBUG && @debug "X-> R..."
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    # Initial step
    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        LogDensity(model, spl),
        0;
        reporter = DynamicHMC.NoProgressReport()
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, stats = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Update the sample.
    vi[spl] = Q.q
    logp = stats.π
    setlogp!(vi, logp)

    return DynamicNUTSTransition(tonamedtuple(vi), logp, Q, steps.H, steps.ϵ)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    ::Integer,
    transition::DynamicNUTSTransition;
    kwargs...
)
    # Compute next sample.
    hamiltonian = transition.hamiltonian
    stepsize = transition.stepsize
    steps = DynamicHMC.MCMCSteps(rng, DynamicHMC.NUTS(), hamiltonian, stepsize)
    Q, stats = DynamicHMC.mcmc_next_step(steps, transition.Q)

    # Update the sample.
    vi = spl.state.vi
    vi[spl] = Q.q
    logp = stats.π
    setlogp!(vi, logp)

    return DynamicNUTSTransition(tonamedtuple(vi), logp, Q, hamiltonian, stepsize)
end

# Do not store fields specific to DynamicHMC.
function AbstractMCMC.transitions_init(
    transition::DynamicNUTSTransition,
    ::Model,
    ::Sampler{<:DynamicNUTS},
    N::Integer;
    kwargs...
)
    return Vector{Transition{typeof(transition.θ),typeof(transition.lp)}}(undef, N)
end

function AbstractMCMC.transitions_save!(
    transitions::Vector{<:Transition},
    iteration::Integer,
    transition::DynamicNUTSTransition,
    ::Model,
    ::Sampler{<:DynamicNUTS},
    ::Integer;
    kwargs...
)
    transitions[iteration] = Transition(transition.θ, transition.lp)
    return
end