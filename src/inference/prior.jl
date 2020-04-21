struct PriorSampler{space} <: InferenceAlgorithm end

PriorSampler() = PriorSampler{()}()

DynamicPPL.getspace(::PriorSampler{space}) where {space} = space

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:PriorSampler},
    ::Integer,
    transition;
    kwargs...
)
    empty!(spl.state.vi)
    model(spl.state.vi, spl)

    return Transition(spl)
end

function Sampler(alg::PriorSampler, model::Model, s::Selector)
    return Sampler(alg, Dict{Symbol, Any}(), s, SamplerState(VarInfo(model)))
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    spl::Sampler{<:PriorSampler},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    return DynamicPPL.assume(SampleFromPrior(), dist, vn, vi)
end

function DynamicPPL.dot_assume(
    spl::Sampler{<:PriorSampler},
    dist::MultivariateDistribution,
    vns::AbstractArray{<:VarName},
    var::AbstractMatrix,
    vi::VarInfo,
)
    return DynamicPPL.dot_assume(SampleFromPrior(), dist, vns, var, vi)
end
function DynamicPPL.dot_assume(
    spl::Sampler{<:PriorSampler},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::VarInfo,
)
    return DynamicPPL.dot_assume(SampleFromPrior(), dist, vns, var, vi)
end

function DynamicPPL.observe(
    spl::Sampler{<:PriorSampler},
    d::Distribution,
    value,
    vi::VarInfo,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:PriorSampler},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end
