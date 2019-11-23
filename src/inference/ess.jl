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

ESSState(model::Model) = ESSState(VarInfo(model))

function Sampler(alg::ESS, model::Model, s::Selector)
    # sanity check
    space = getspace(alg)
    if isempty(space)
        pvars = get_pvars(model)
        length(pvars) == 1 ||
            error("[ESS] no symbol specified to sampler although there is not exactly one model parameter ($pvars)")
    end

    state = ESSState(model)
    info = Dict{Symbol, Any}()

    return Sampler(alg, info, s, state)
end

# always accept in the first step
function step!(::AbstractRNG, ::Model, spl::Sampler{<:ESS}, ::Integer; kwargs...)
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
    # recompute log-likelihood in logp
    vi = spl.state.vi
    if spl.selector.tag !== :default
        runmodel!(model, vi, spl)
    end

    # obtain previous sample
    f = copy(vi[spl])

    # sample log-likelihood threshold for the next sample
    threshold = getlogp(vi) - randexp(rng)

    # sample from the prior
    runmodel!(model, vi, spl)
    ν = copy(vi[spl])

    # sample initial angle
    θ = 2 * π * rand(rng)
    θₘᵢₙ = θ - 2 * π
    θₘₐₓ = θ

    while true
        # compute proposal
        sinθ, cosθ = sincos(θ)
        @. vi[spl] = f * cosθ + ν * sinθ

        # recompute log-likelihood and check if threshold is reached
        resetlogp!(vi)
        model(vi, spl)
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

isnormal(dist) = false
isnormal(::Normal) = true
isnormal(::NormalCanon) = true
isnormal(::AbstractMvNormal) = true

function assume(spl::Sampler{<:ESS}, dist::Distribution, vn::VarName, vi::VarInfo)
    space = getspace(spl)
    if space === () || space === (vn.sym,)
        isnormal(dist) ||
            error("[ESS] does only support normally distributed prior distributions")

        r = rand(dist)
        vi[vn] = vectorize(dist, r)
        setgid!(vi, spl.selector, vn)
        return r, zero(Base.promote_eltype(dist, r))
    else
        r = vi[vn]
        return r, logpdf(dist, r)
    end
end

function observe(spl::Sampler{<:ESS}, dist::Distribution, value, vi::VarInfo)
    return observe(dist, value, vi)
end
