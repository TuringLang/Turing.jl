module TuringDistributionsExt

#
# Random probability measures as distributions
#

using Distributions
using StatsFuns: logsumexp, softmax!

import Distributions: logpdf
import Base: maximum, minimum, rand
import Random: AbstractRNG

# Access these types through `Base.get_extension(Turing, :TuringDistributionsExt)`.

#
# Representations
#

abstract type AbstractRandomProbabilityMeasure end

"""
    SizeBiasedSamplingProcess(rpm, surplus)

A size-biased sampling representation of the random probability measure `rpm` with remaining
mass `surplus`.
"""
struct SizeBiasedSamplingProcess{T<:AbstractRandomProbabilityMeasure,V<:AbstractFloat} <:
       ContinuousUnivariateDistribution
    rpm::T
    surplus::V
end

logpdf(d::SizeBiasedSamplingProcess, x::Real) = logpdf(distribution(d), x)
rand(rng::AbstractRNG, d::SizeBiasedSamplingProcess) = rand(rng, distribution(d))
minimum(d::SizeBiasedSamplingProcess) = zero(d.surplus)
maximum(d::SizeBiasedSamplingProcess) = d.surplus

"""
    StickBreakingProcess(rpm)

A stick-breaking representation of the random probability measure `rpm`.
"""
struct StickBreakingProcess{T<:AbstractRandomProbabilityMeasure} <:
       ContinuousUnivariateDistribution
    rpm::T
end

logpdf(d::StickBreakingProcess, x::Real) = logpdf(distribution(d), x)
rand(rng::AbstractRNG, d::StickBreakingProcess) = rand(rng, distribution(d))
minimum(::StickBreakingProcess) = 0.0
maximum(::StickBreakingProcess) = 1.0

"""
    ChineseRestaurantProcess(rpm, m)

A Chinese restaurant process for the random probability measure `rpm`, where `m` contains the
current cluster counts.
"""
struct ChineseRestaurantProcess{
    T<:AbstractRandomProbabilityMeasure,V<:AbstractVector{<:Integer}
} <: DiscreteUnivariateDistribution
    rpm::T
    m::V
end

function _logpdf_table end

function logpdf(d::ChineseRestaurantProcess, x::Integer)
    if insupport(d, x)
        logweights = _logpdf_table(d.rpm, d.m)
        return logweights[x] - logsumexp(logweights)
    else
        return -Inf
    end
end

function rand(rng::AbstractRNG, d::ChineseRestaurantProcess)
    logweights = _logpdf_table(d.rpm, d.m)
    softmax!(logweights)
    return rand(rng, Categorical(logweights))
end

minimum(::ChineseRestaurantProcess) = 1
function maximum(d::ChineseRestaurantProcess)
    return any(iszero, d.m) ? length(d.m) : length(d.m) + 1
end

#
# Dirichlet process
#

"""
    DirichletProcess(α)

A Dirichlet process with concentration parameter `α`.

Its size-biased and stick-breaking representations draw proportions according to

```math
V_k \\sim \\operatorname{Beta}(1, \\alpha).
```

In the Chinese restaurant representation, an occupied cluster `k` has weight `m[k]`, while a
new cluster has weight `α`.

# References

Yee Whye Teh, "Dirichlet Process," 2010.
https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf
"""
struct DirichletProcess{T<:Real} <: AbstractRandomProbabilityMeasure
    α::T
end

function distribution(d::StickBreakingProcess{<:DirichletProcess})
    α = d.rpm.α
    return Beta(one(α), α)
end

function distribution(d::SizeBiasedSamplingProcess{<:DirichletProcess})
    α = d.rpm.α
    return LocationScale(zero(α), d.surplus, Beta(one(α), α))
end

function _logpdf_table(d::DirichletProcess, m::AbstractVector{<:Integer})
    first_zero = findfirst(iszero, m)
    ntables = first_zero === nothing ? length(m) + 1 : length(m)
    new_cluster_logweight = log(d.α)
    table = fill(oftype(new_cluster_logweight, -Inf), ntables)

    if iszero(m)
        table[1] = zero(new_cluster_logweight)
        return table
    end

    @inbounds for i in eachindex(m)
        !iszero(m[i]) && (table[i] = log(m[i]))
    end

    new_cluster = first_zero === nothing ? ntables : first_zero
    table[new_cluster] = new_cluster_logweight
    return table
end

#
# Pitman-Yor process
#

"""
    PitmanYorProcess(d, θ, t)

A Pitman-Yor process with discount `d`, concentration `θ`, and `t` occupied clusters.

Its size-biased and stick-breaking representations draw proportions according to

```math
V_k \\sim \\operatorname{Beta}(1 - d, \\theta + t d).
```

In the Chinese restaurant representation, an occupied cluster `k` has weight `m[k] - d`, while
a new cluster has weight `θ + d * t`.

# References

Jim Pitman and Marc Yor, "The two-parameter Poisson-Dirichlet distribution derived from a stable
subordinator," 1997.
"""
struct PitmanYorProcess{T<:Real,I<:Integer} <: AbstractRandomProbabilityMeasure
    d::T
    θ::T
    t::I
end

function distribution(d::StickBreakingProcess{<:PitmanYorProcess})
    rpm = d.rpm
    discount = rpm.d
    return Beta(one(discount) - discount, rpm.θ + rpm.t * discount)
end

function distribution(d::SizeBiasedSamplingProcess{<:PitmanYorProcess})
    rpm = d.rpm
    discount = rpm.d
    dist = Beta(one(discount) - discount, rpm.θ + rpm.t * discount)
    return LocationScale(zero(discount), d.surplus, dist)
end

function _logpdf_table(d::PitmanYorProcess, m::AbstractVector{<:Integer})
    @assert d.t == count(!iszero, m)

    first_zero = findfirst(iszero, m)
    ntables = first_zero === nothing ? length(m) + 1 : length(m)
    new_cluster_logweight = log(d.θ + d.d * d.t)
    table = fill(oftype(new_cluster_logweight, -Inf), ntables)

    if iszero(m)
        table[1] = zero(new_cluster_logweight)
        return table
    end

    @inbounds for i in eachindex(m)
        !iszero(m[i]) && (table[i] = log(m[i] - d.d))
    end

    new_cluster = first_zero === nothing ? ntables : first_zero
    table[new_cluster] = new_cluster_logweight
    return table
end

#
# Stick breaking
#

"""
    stickbreak(v)

Convert `K - 1` breaking proportions in `v` into `K` simplex weights.
"""
function stickbreak(v)
    isempty(v) && return [one(eltype(v))]

    K = length(v) + 1
    remaining = cumprod(1 .- v)
    return [
        if k == 1
            v[1]
        elseif k == K
            remaining[K - 1]
        else
            v[k] * remaining[k - 1]
        end for k in 1:K
    ]
end

end
