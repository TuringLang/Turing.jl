###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###
struct DynamicNUTS{AD, space} <: Hamiltonian{AD} end

using LogDensityProblems: LogDensityProblems

struct FunctionLogDensity{F}
    dimension::Int
    f::F
end

LogDensityProblems.dimension(ℓ::FunctionLogDensity) = ℓ.dimension

function LogDensityProblems.capabilities(::Type{<:FunctionLogDensity})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity(ℓ::FunctionLogDensity, x::AbstractVector)
    first(ℓ.f(x))
end

function LogDensityProblems.logdensity_and_gradient(ℓ::FunctionLogDensity,
                                                    x::AbstractVector)
    ℓ.f(x)
end

"""
    DynamicNUTS()

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package. To use it, make
sure you have the DynamicHMC package (version `2.*`) loaded:

```julia
using DynamicHMC
``
"""
DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
DynamicNUTS{AD}() where AD = DynamicNUTS{AD, ()}()
function DynamicNUTS{AD}(space::Symbol...) where AD
    DynamicNUTS{AD, space}()
end

struct DynamicNUTSState{V<:AbstractVarInfo,D}
    vi::V
    draws::Vector{D}
end

DynamicPPL.getspace(::DynamicNUTS{<:Any, space}) where {space} = space

DynamicPPL.initialsampler(::Sampler{<:DynamicNUTS}) = SampleFromUniform()

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    vi::AbstractVarInfo;
    N::Int,
    kwargs...
)
    # Set up lp function.
    function _lp(x)
        gradient_logp(x, vi, model, spl)
    end

    link!(vi, spl)
    l, dl = _lp(vi[spl])
    while !isfinite(l) || !isfinite(dl)
        model(vi, SampleFromUniform())
        link!(vi, spl)
        l, dl = _lp(vi[spl])
    end

    if spl.selector.tag == :default && !islinked(vi, spl)
        link!(vi, spl)
        model(vi, spl)
    end

    results = mcmc_with_warmup(
        rng,
        FunctionLogDensity(
            length(vi[spl]),
            _lp
        ),
        N
    )
    draws = results.chain

    # Compute first transition and state.
    draw = popfirst!(draws)
    vi[spl] = draw
    transition = Transition(vi)
    state = DynamicNUTSState(vi, draws)

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    state::DynamicNUTSState;
    kwargs...
)
    # Extract VarInfo object.
    vi = state.vi

    # Pop the next draw off the vector.
    draw = popfirst!(state.draws)
    vi[spl] = draw

    # Compute next transition.
    transition = Transition(vi)

    return transition, state
end

# Disable the progress logging for DynamicHMC, since it has its own progress meter.
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::DynamicNUTS,
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    if progress
        @warn "[HMC] Progress logging in Turing is disabled since DynamicHMC provides its own progress meter"
    end
    if resume_from === nothing
        return AbstractMCMC.sample(rng, model, Sampler(alg, model), N;
                                   chain_type=chain_type, progress=false, N=N, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=false, N=N, kwargs...)
    end
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::DynamicNUTS,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    if progress
        @warn "[HMC] Progress logging in Turing is disabled since DynamicHMC provides its own progress meter"
    end
    return AbstractMCMC.sample(rng, model, Sampler(alg, model), parallel, N, n_chains;
                               chain_type=chain_type, progress=false, N=N, kwargs...)
end
