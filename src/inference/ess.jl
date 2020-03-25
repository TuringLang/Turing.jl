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
        isgaussian(dist) ||
            error("[ESS] only supports Gaussian prior distributions")
    end

    state = ESSState(vi)
    info = Dict{Symbol, Any}()

    return Sampler(alg, info, s, state)
end

isgaussian(dist) = false
isgaussian(::Normal) = true
isgaussian(::NormalCanon) = true
isgaussian(::AbstractMvNormal) = true

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
    # obtain mean of distribution
    vi = spl.state.vi
    vns = _getvns(vi, spl)
    μ = mapreduce(vcat, vns[1]) do vn
        dist = getdist(vi, vn)
        vectorize(dist, mean(dist))
    end

    # obtain previous sample
    f = vi[spl]

    # recompute log-likelihood in logp
    if spl.selector.tag !== :default
        runmodel!(model, vi, spl)
    end

    # sample log-likelihood threshold for the next sample
    threshold = getlogp(vi) - randexp(rng)

    # sample from the prior
    set_flag!(vi, vns[1][1], "del")
    runmodel!(model, vi, spl)
    ν = vi[spl]

    # sample initial angle
    θ = 2 * π * rand(rng)
    θmin = θ - 2 * π
    θmax = θ

    while true
        # compute proposal and apply correction for distributions with nonzero mean
        sinθ, cosθ = sincos(θ)
        a = 1 - (sinθ + cosθ)
        vi[spl] = @. f * cosθ + ν * sinθ + μ * a

        # recompute log-likelihood and check if threshold is reached
        runmodel!(model, vi, spl)
        if getlogp(vi) > threshold
            break
        end

        # shrink the bracket
        if θ < 0
            θmin = θ
        else
            θmax = θ
        end

        # sample new angle
        θ = θmin + rand(rng) * (θmax - θmin)
    end

    return Transition(spl)
end

function tilde(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, vn::VarName, inds, vi)
    if vn in getspace(sampler)
        return tilde(LikelihoodContext(), SampleFromPrior(), right, vn, inds, vi)
    else
        return tilde(ctx, SampleFromPrior(), right, vn, inds, vi)
    end
end

function tilde(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return tilde(ctx, SampleFromPrior(), right, left, vi)
end

function dot_tilde(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vn::VarName, inds, vi)
    if vn in getspace(sampler)
        return dot_tilde(LikelihoodContext(), SampleFromPrior(), right, left, vn, inds, vi)
    else
        return dot_tilde(ctx, SampleFromPrior(), right, left, vn, inds, vi)
    end
end

function dot_tilde(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return dot_tilde(ctx, SampleFromPrior(), right, left, vi)
end