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

mutable struct DynamicNUTSState{V<:VarInfo, D} <: AbstractSamplerState
    vi::V
    draws::Vector{D}
end

DynamicPPL.getspace(::DynamicNUTS{<:Any, space}) where {space} = space

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    N::Integer;
    kwargs...
)
    # Set up lp function.
    function _lp(x)
        gradient_logp(x, spl.state.vi, model, spl)
    end

    # Set the parameters to a starting value.
    initialize_parameters!(spl; kwargs...)

    model(spl.state.vi, SampleFromUniform())
    link!(spl.state.vi, spl)
    l, dl = _lp(spl.state.vi[spl])
    while !isfinite(l) || !isfinite(dl)
        model(spl.state.vi, SampleFromUniform())
        link!(spl.state.vi, spl)
        l, dl = _lp(spl.state.vi[spl])
    end

    if spl.selector.tag == :default && !islinked(spl.state.vi, spl)
        link!(spl.state.vi, spl)
        model(spl.state.vi, spl)
    end

    results = mcmc_with_warmup(
        rng,
        FunctionLogDensity(
            length(spl.state.vi[spl]),
            _lp
        ),
        N
    )

    spl.state.draws = results.chain
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    N::Integer,
    transition;
    kwargs...
)
    # Pop the next draw off the vector.
    draw = popfirst!(spl.state.draws)
    spl.state.vi[spl] = draw
    return Transition(spl)
end

function Sampler(
    alg::DynamicNUTS,
    model::Model,
    s::Selector=Selector()
)
    # Construct a state, using a default function.
    state = DynamicNUTSState(VarInfo(model), [])

    # Return a new sampler.
    return Sampler(alg, Dict{Symbol,Any}(), s, state)
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
                                   chain_type=chain_type, progress=false, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=false, kwargs...)
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
                               chain_type=chain_type, progress=false, kwargs...)
end
