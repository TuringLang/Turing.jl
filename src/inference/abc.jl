"""
doesn't work yet
more or less copy pasted IS which is quite similar to ABC
"""
struct ABC{space} <: InferenceAlgorithm end

ABC() = ABC{()}()

# perhaps we just need vi
mutable struct ABCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                 ::  V
    final_logevidence  ::  F
end

ABCState(model::Model) = ABCState(VarInfo(model), 0.0)

function Sampler(alg::ABC, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ABCState(model)
    return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ABC},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.vi)
    model(rng, spl.state.vi, spl)

    return Transition(spl)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ABC},
    N::Integer,
    ts::Vector;
    kwargs...
)
    # Calculate evidence.
    spl.state.final_logevidence = logsumexp(map(x->x.lp, ts)) - log(N)
end

function DynamicPPL.assume(rng, spl::Sampler{<:ABC}, dist::Distribution, vn::VarName, vi)
    r = rand(rng, dist)
    push!(vi, vn, r, dist, spl)
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:ABC}, dist::Distribution, value, vi)
    # acclogp!(vi, logpdf(dist, value))
    return logpdf(dist, value)
end