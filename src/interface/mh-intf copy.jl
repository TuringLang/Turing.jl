using Turing.Interface
using Statistics
using Distributions
import Base.+
import Distributions: Sampleable, GLOBAL_RNG

mutable struct Model{DT,L} <: Sampleable{Distributions.Univariate, Distributions.Continuous}
    data :: Vector{DT}
    likelihood :: L
end

struct Transition <: AbstractTransition
    μ :: Float64
    σ :: Float64
end

Base.:+(t1::Transition, t2::Transition) = Transition(t1.μ + t2.μ, t1.σ + t2.σ)

mutable struct MH <: AbstractSampler end
transition_type(s::MH) = Transition

propose(ℓ::Model) = Transition(mean(ℓ.data),std(ℓ.data))
propose(θ) = Transition(rand(gμ), rand(gσ)) + θ

function acceptance(θ::T, θ_new::T) where T<:AbstractTransition
    fn = fieldnames(T)
    probs = Vector{Float64}(undef, length(fn))
    for i in eachindex(fn)
        e = getproperty(θ, fn[i])
        e_new = getproperty(θ_new, fn[i])
        g = Normal(e, 1)
        g_new = Normal(e_new, 1)
        probs[i] = logpdf(e, gσ_new) / logpdf(e_new, e)
    end
    return probs
end

function Interface.step!(
    rng::AbstractRNG,
    ℓ::Model,
    s::MH,
    N::Integer;
    kwargs...
)
    return propose()
end

function Interface.step!(
    rng::AbstractRNG,
    ℓ::Model,
    s::MH,
    N::Integer,
    θ::Transition;
    kwargs...
)
    θ_new, trans_prob = propose(θ)

    # The support is violated, reject the sample.
    if θ_new.σ <= 0
        return θ
    end

    
    l_new = ℓ.likelihood(ℓ, θ_new)
    l_old = ℓ.likelihood(ℓ, θ)

    # println(exp(l_new))
    # println(exp(l_old))

    α = min(1.0, l_new / l_old)
    # println("""
    # θ     = $θ
    # θ'    = $θ_new
    # 𝓛(θ)  = $l_old
    # 𝓛(θ') = $l_new
    # α     = $α
    # """)
    if rand() < α
        return θ_new
    else
        return θ
    end
end

function Chains(rng, ℓ::Model, s::MH, N, ts::Vector{T}; kwargs...) where {T<:AbstractTransition}
    fields = [f for f in fieldnames(T)]
    vals = [[getproperty(t, f) for f in fields] for t in ts]
    return Chains(vals, string.(fields))
end

dist = Normal(15, 1)
obs = rand(dist, 100)

function likelihood(ℓ, θ)
    d = Normal(θ.μ, θ.σ)
    return loglikelihood(d, ℓ.data)
end

ℓ = Model(obs, likelihood)

chain = sample(ℓ, MH(), 100000)

import Turing

Turing.@model gdemo(xs) = begin
    μ ~ Normal(0, 1)
    σ ~ TruncatedNormal(0, 1, 0, Inf)
    for i in 1:length(xs)
        xs[i] ~ Normal(μ, σ)
    end
end

# chain2 = sample(gdemo(obs), Turing.MH(), 100000)