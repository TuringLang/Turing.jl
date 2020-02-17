module RandomMeasures

#using ..Utilities
using Distributions
using LinearAlgebra
using StatsFuns: logsumexp, softmax!

import Distributions: logpdf
import Base: maximum, minimum, rand
import Random: AbstractRNG

## ############### ##
## Representations ##
## ############### ##

abstract type AbstractRandomProbabilityMeasure end

"""
    SizeBiasedSamplingProcess(rpm, surplus)

The *Size-Biased Sampling Process* for random probability measures `rpm` with a surplus mass of `surplus`.
"""
struct SizeBiasedSamplingProcess{T<:AbstractRandomProbabilityMeasure,V<:AbstractFloat} <: ContinuousUnivariateDistribution
    rpm::T
    surplus::V
end

logpdf(d::SizeBiasedSamplingProcess, x::Real) = logpdf(distribution(d), x)
rand(rng::AbstractRNG, d::SizeBiasedSamplingProcess) = rand(rng, distribution(d))
minimum(d::SizeBiasedSamplingProcess) = zero(d.surplus)
maximum(d::SizeBiasedSamplingProcess) = d.surplus

"""
    StickBreakingProcess(rpm)

The *Stick-Breaking Process* for random probability measures `rpm`.
"""
struct StickBreakingProcess{T<:AbstractRandomProbabilityMeasure} <: ContinuousUnivariateDistribution
    rpm::T
end

logpdf(d::StickBreakingProcess, x::Real) = logpdf(distribution(d), x)
rand(rng::AbstractRNG, d::StickBreakingProcess) = rand(rng, distribution(d))
minimum(d::StickBreakingProcess) = 0.0
maximum(d::StickBreakingProcess) = 1.0

"""
    ChineseRestaurantProcess(rpm, m)

The *Chinese Restaurant Process* for random probability measures `rpm` with counts `m`.
"""
struct ChineseRestaurantProcess{T<:AbstractRandomProbabilityMeasure,V<:AbstractVector{Int}} <: DiscreteUnivariateDistribution
    rpm::T
    m::V
end


"""
    _logpdf_table(d::AbstractRandomProbabilityMeasure, m::AbstractVector{Int})

Parameters:

* `d`: Random probability measure, e.g. DirichletProcess
* `m`: Cluster counts

"""
function _logpdf_table end

function logpdf(d::ChineseRestaurantProcess, x::Int)
    if insupport(d, x)
        lp = _logpdf_table(d.rpm, d.m)
        return lp[x] - logsumexp(lp)
    else
        return -Inf
    end
end

function rand(rng::AbstractRNG, d::ChineseRestaurantProcess)
    lp = _logpdf_table(d.rpm, d.m)
    softmax!(lp)
    return rand(rng, Categorical(lp))
end

minimum(d::ChineseRestaurantProcess) = 1
maximum(d::ChineseRestaurantProcess) = any(iszero, d.m) ? length(d.m) : length(d.m)+1

## ################# ##
## Random partitions ##
## ################# ##

"""
    DirichletProcess(α)

The *Dirichlet Process* with concentration parameter `α`.
Samples from the Dirichlet process can be constructed via the following representations.

*Size-Biased Sampling Process*
```math
j_k \\sim Beta(1, \\alpha) * surplus
```

*Stick-Breaking Process*
```math
v_k \\sim Beta(1, \\alpha)
```

*Chinese Restaurant Process*
```math
p(z_n = k | z_{1:n-1}) \\propto \\begin{cases}
        \\frac{m_k}{n-1+\\alpha}, \\text{if} m_k > 0\\\\
        \\frac{\\alpha}{n-1+\\alpha}
    \\end{cases}
```

For more details see: https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf
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

function _logpdf_table(d::DirichletProcess{T}, m::AbstractVector{Int}) where {T<:Real}

    # construct the table
    first_zero = findfirst(iszero, m)
    K = first_zero === nothing ? length(m)+1 : length(m)
    table = fill(T(-Inf), K)

    # exit if m is empty or contains only zeros
    if iszero(m)
        table[1] = T(0)
        return table
    end

    # compute logpdf for each occupied table
    @inbounds for i in 1:(K-1)
        table[i] = T(log(m[i]))
    end

    # logpdf for new table
    k_new = first_zero === nothing ? K : first_zero
    table[k_new] = log(d.α)

    return table
end

"""
    PitmanYorProcess(d, θ, t)

The *Pitman-Yor Process* with discount `d`, concentration `θ` and `t` already drawn atoms.
Samples from the *Pitman-Yor Process* can be constructed via the following representations.

*Size-Biased Sampling Process*
```math
j_k \\sim Beta(1-d, \\theta + t*d) * surplus
```

*Stick-Breaking Process*
```math
v_k \\sim Beta(1-d, \\theta + t*d)
```

*Chinese Restaurant Process*
```math
p(z_n = k | z_{1:n-1}) \\propto \\begin{cases}
        \\frac{m_k - d}{n+\\theta}, \\text{if} m_k > 0\\\\
        \\frac{\\theta + d*t}{n+\\theta}
    \\end{cases}
```

For more details see: https://en.wikipedia.org/wiki/Pitman–Yor_process
"""
struct PitmanYorProcess{T<:Real} <: AbstractRandomProbabilityMeasure
    d::T
    θ::T
    t::Int
end

function distribution(d::StickBreakingProcess{<:PitmanYorProcess})
    d_rpm = d.rpm
    d_rpm_d = d.rpm.d
    return Beta(one(d_rpm_d)-d_rpm_d, d_rpm.θ + d_rpm.t*d_rpm_d)
end

function distribution(d::SizeBiasedSamplingProcess{<:PitmanYorProcess})
    d_rpm = d.rpm
    d_rpm_d = d.rpm.d
    dist = Beta(one(d_rpm_d)-d_rpm_d, d_rpm.θ + d_rpm.t*d_rpm_d)
    return LocationScale(zero(d_rpm_d), d.surplus, dist)
end

function _logpdf_table(d::PitmanYorProcess{T}, m::AbstractVector{Int}) where {T<:Real}
    # sanity check
    @assert d.t == sum(!iszero, m)

    # construct table
    first_zero = findfirst(iszero, m)
    K = first_zero === nothing ? length(m)+1 : length(m)
    table = fill(T(-Inf), K)

    # exit if m is empty or contains only zeros
    if iszero(m)
        table[1] = T(0)
        return table
    end

    # compute logpdf for each occupied table
    @inbounds for i in 1:(K-1)
        !iszero(m[i]) && ( table[i] = T(log(m[i] - d.d)) )
    end

    # logpdf for new table
    k_new = first_zero === nothing ? K : first_zero
    table[k_new] = log(d.θ + d.d * d.t)

    return table
end

## ####### ##
## Exports ##
## ####### ##

export DirichletProcess, PitmanYorProcess
export SizeBiasedSamplingProcess, StickBreakingProcess, ChineseRestaurantProcess


end # end module
