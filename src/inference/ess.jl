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
│ 1   │ m          │ 0.811555 │
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
    dist = getdist(vi, vns[1][1])
    isgaussian(dist) ||
        error("[ESS] only supports Gaussian prior distributions")

    state = ESSState(vi)
    info = Dict{Symbol, Any}()

    return Sampler(alg, info, s, state)
end

isgaussian(dist) = false
isgaussian(::Normal) = true
isgaussian(::NormalCanon) = true
isgaussian(::AbstractMvNormal) = true

# always accept in the first step
function step!(::AbstractRNG, model::Model, spl::Sampler{<:ESS}, ::Integer; kwargs...)
    return Transition(spl)
end

function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ESS},
    ::Integer,
    ::Transition;
    kwargs...
)
    # obtain mean of distribution
    vi = spl.state.vi
    vn = _getvns(vi, spl)[1][1]
    dist = getdist(vi, vn)
    μ = vectorize(dist, mean(dist))

    # obtain previous sample
    f = vi[vn]

    # recompute log-likelihood in logp
    if spl.selector.tag !== :default
        runmodel!(model, vi, spl)
    end
    setgid!(vi, spl.selector, vn)

    # sample log-likelihood threshold for the next sample
    threshold = getlogp(vi) - randexp(rng)

    # sample from the prior
    ν = vectorize(dist, rand(rng, dist))

    # sample initial angle
    θ = 2 * π * rand(rng)
    θₘᵢₙ = θ - 2 * π
    θₘₐₓ = θ

    while true
        # compute proposal and apply correction for distributions with nonzero mean
        sinθ, cosθ = sincos(θ)
        a = 1 - (sinθ + cosθ)
        vi[vn] = @. f * cosθ + ν * sinθ + μ * a

        # recompute log-likelihood and check if threshold is reached
        runmodel!(model, vi, spl)
        if getlogp(vi) > threshold
            break
        end

        # shrink the bracket
        if θ < 0
            θₘᵢₙ = θ
        else
            θₘₐₓ = θ
        end

        # sample new angle
        θ = θₘᵢₙ + rand(rng) * (θₘₐₓ - θₘᵢₙ)
    end

    return Transition(spl)
end

function assume(spl::Sampler{<:ESS}, dist::Distribution, vn::VarName, vi::VarInfo)
    # don't sample
    r = vi[vn]

    # avoid possibly costly computation of the prior probability
    space = getspace(spl)
    if space === () || space === (vn.sym,)
        return r, zero(Base.promote_eltype(dist, r))
    else
        return r, logpdf_with_trans(dist, r, istrans(vi, vn))
    end
end

function observe(spl::Sampler{<:ESS}, dist::Distribution, value, vi::VarInfo)
    return observe(dist, value, vi)
end
