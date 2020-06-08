"""
    ESS

Elliptical slice sampling algorithm.

# Examples
```jldoctest; setup = :(Random.seed!(1))
julia> @model gdemo(x) = begin
           m ~ Normal()
           x ~ Normal(m, 0.5)
       end
gdemo (generic function with 2 methods)

julia> sample(gdemo(1.0), ESS(), 1_000) |> mean
Mean

│ Row │ parameters │ mean     │
│     │ Symbol     │ Float64  │
├─────┼────────────┼──────────┤
│ 1   │ m          │ 0.824853 │
```
"""
struct ESS{space} <: InferenceAlgorithm end

ESS() = ESS{()}()
ESS(space::Symbol) = ESS{(space,)}()

isgibbscomponent(::ESS) = true

mutable struct ESSState{V<:VarInfo} <: AbstractSamplerState
    vi::V
end

function Sampler(alg::ESS, model::Model, s::Selector)
    # sanity check
    vi = VarInfo(model)
    space = getspace(alg)
    vns = _getvns(vi, s, Val(space))
    length(vns) == 1 ||
        error("[ESS] does only support one variable ($(length(vns)) variables specified)")
    for vn in vns[1]
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("[ESS] only supports Gaussian prior distributions")
    end

    state = ESSState(vi)
    info = Dict{Symbol, Any}()

    return Sampler(alg, info, s, state)
end

# always accept in the first step
function AbstractMCMC.step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:ESS},
    ::Integer,
    ::Nothing;
    kwargs...
)
    return Transition(spl)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ESS},
    ::Integer,
    transition;
    kwargs...
)
    # obtain random variable that should be sampled
    vi = spl.state.vi

    # obtain previous sample
    f = vi[spl]

    # recompute log-likelihood in logp
    if spl.selector.tag !== :default
        model(vi, spl)
    end

    # define previous sampler state
    oldstate = EllipticalSliceSampling.EllipticalSliceSamplerState(f, getlogp(vi))

    # compute next state
    state = AbstractMCMC.step!(rng, ESSModel(model, spl),
                               EllipticalSliceSampling.EllipticalSliceSampler(), 1, oldstate;
                               progress = false)

    # update sample and log-likelihood
    vi[spl] = state.sample
    setlogp!(vi, state.loglikelihood)

    return Transition(spl)
end

struct ESSModel{M<:Model,S<:Sampler{<:ESS},T} <: AbstractMCMC.AbstractModel
    model::M
    spl::S
    μ::T
end

function ESSModel(model::Model, spl::Sampler{<:ESS})
    vi = spl.state.vi
    vns = _getvns(vi, spl)
    μ = mapreduce(vcat, vns[1]) do vn
        dist = getdist(vi, vn)
        vectorize(dist, mean(dist))
    end

    ESSModel(model, spl, μ)
end

# sample from the prior
function EllipticalSliceSampling.sample_prior(rng::Random.AbstractRNG, model::ESSModel)
    spl = model.spl
    vi = spl.state.vi
    vns = _getvns(vi, spl)
    set_flag!(vi, vns[1][1], "del")
    model.model(rng, vi, spl)
    return vi[spl]
end

# compute proposal and apply correction for distributions with nonzero mean
function EllipticalSliceSampling.proposal(model::ESSModel, f, ν, θ)
    sinθ, cosθ = sincos(θ)
    a = 1 - (sinθ + cosθ)
    return @. cosθ * f + sinθ * ν + a * model.μ
end

function EllipticalSliceSampling.proposal!(out, model::ESSModel, f, ν, θ)
    sinθ, cosθ = sincos(θ)
    a = 1 - (sinθ + cosθ)
    @. out = cosθ * f + sinθ * ν + a * model.μ
    return out
end

# evaluate log-likelihood
function Distributions.loglikelihood(model::ESSModel, f)
    spl = model.spl
    vi = spl.state.vi
    vi[spl] = f
    model.model(vi, spl)
    getlogp(vi)
end

function DynamicPPL.tilde(rng, ctx::DefaultContext, sampler::Sampler{<:ESS}, right, vn::VarName, inds, vi)
    if inspace(vn, sampler)
        return DynamicPPL.tilde(rng, LikelihoodContext(), SampleFromPrior(), right, vn, inds, vi)
    else
        return DynamicPPL.tilde(rng, ctx, SampleFromPrior(), right, vn, inds, vi)
    end
end

function DynamicPPL.tilde(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return DynamicPPL.tilde(ctx, SampleFromPrior(), right, left, vi)
end

function DynamicPPL.dot_tilde(rng, ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vn::VarName, inds, vi)
    if inspace(vn, sampler)
        return DynamicPPL.dot_tilde(rng, LikelihoodContext(), SampleFromPrior(), right, left, vn, inds, vi)
    else
        return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vn, inds, vi)
    end
end

function DynamicPPL.dot_tilde(rng, ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vi)
end
