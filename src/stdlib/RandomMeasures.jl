module RandomMeasures

using ..Core, ..Core.RandomVariables, ..Utilities
using Distributions
using LinearAlgebra
using StatsFuns: logsumexp

import Distributions: sample, logpdf
import Base: maximum, minimum, rand

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

logpdf(d::SizeBiasedSamplingProcess, x) = _logpdf(d, x)
rand(d::SizeBiasedSamplingProcess) = _rand(d)
minimum(d::SizeBiasedSamplingProcess) = zero(d.surplus)
maximum(d::SizeBiasedSamplingProcess) = d.surplus

"""
    StickBreakingProcess(rpm)

The *Stick-Breaking Process* for random probability measures `rpm`.
"""
struct StickBreakingProcess{T<:AbstractRandomProbabilityMeasure} <: ContinuousUnivariateDistribution
    rpm::T
end

logpdf(d::StickBreakingProcess, x) = _logpdf(d, x)
rand(d::StickBreakingProcess) = _rand(d)
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
    _logpdf_table(d<:AbstractRandomProbabilityMeasure, m<:AbstractVector{Int})

Parameters:

* `d`: Random probability measure, e.g. DirichletProcess
* `m`: Cluster counts

"""
function _logpdf_table(d::AbstractRandomProbabilityMeasure, m::T) where {T<:AbstractVector{Int}}
    throw(MethodError(_logpdf_table(), (d, m)))
end

function logpdf(d::ChineseRestaurantProcess, x::Int)
    if insupport(d, x)
        lp = _logpdf_table(d.rpm, d.m)
        return lp[x] - logsumexp(lp)
    else
        return -Inf
    end
end

function rand(d::ChineseRestaurantProcess)
    lp = _logpdf_table(d.rpm, d.m)
    p = exp.(lp)
    return rand(Categorical(p ./ sum(p)))
end

minimum(d::ChineseRestaurantProcess) = 1
maximum(d::ChineseRestaurantProcess) = length(d.m) + 1

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

_rand(d::StickBreakingProcess{DirichletProcess{T}}) where {T<:Real} = rand(Beta(one(T), d.rpm.α))

function _rand(d::SizeBiasedSamplingProcess{DirichletProcess{T}}) where {T<:Real}
    return d.surplus*rand(Beta(one(T), d.rpm.α))
end

function _logpdf(d::StickBreakingProcess{DirichletProcess{T}}, x::T) where {T<:Real}
    return logpdf(Beta(one(T), d.rpm.α), x)
end

function _logpdf(d::SizeBiasedSamplingProcess{DirichletProcess{T}}, x::T) where {T<:Real}
    return logpdf(Beta(one(T), d.rpm.α), x/d.surplus)
end

function _logpdf_table(d::DirichletProcess{V}, m::T) where {T<:AbstractVector{Int},V<:Real}
    if sum(m) == 0
        return zeros(V,1)
    elseif any(m_ -> m_ == 0, m)
        z = log(sum(m) - 1 + d.α)
        K = length(m)
        zid = findfirst(m_ -> m_ == 0, m)
        lpt(k) = k == zid ? log(d.α) - z : log(m[k]) - z
        return map(k -> lpt(k), 1:K)
    else
        z = log(sum(m) - 1 + d.α)
        K = length(m)
        lp(k) = k > K ? log(d.α) - z : log(m[k]) - z
        return map(k -> lp(k), 1:(K+1))
    end
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

function _rand(d::StickBreakingProcess{PitmanYorProcess{T}}) where {T<:Real}
    return rand(Beta(one(T)-d.rpm.d, d.rpm.θ + d.rpm.t*d.rpm.d))
end

function _rand(d::SizeBiasedSamplingProcess{PitmanYorProcess{T}}) where {T<:Real}
    return d.surplus*rand(Beta(one(T)-d.rpm.d, d.rpm.θ + d.rpm.t*d.rpm.d))
end

function _logpdf(d::StickBreakingProcess{PitmanYorProcess{T}}, x::T) where {T<:Real}
    return logpdf(Beta(one(V)-d.rpm.d, d.rpm.θ + d.rpm.t*d.rpm.d), x)
end

function _logpdf(d::SizeBiasedSamplingProcess{PitmanYorProcess{T}}, x::T) where {T<:Real}
    return logpdf(Beta(one(V)-d.rpm.d, d.rpm.θ + d.rpm.t*d.rpm.d), x/d.surplus)
end

function _logpdf_table(d::PitmanYorProcess{V}, m::T) where {T<:AbstractVector{Int},V<:Real}
    if sum(m) == 0
        return zeros(V,1)
    elseif any(m_ -> m_ == 0, m)
        z = log(sum(m) + d.θ)
        K = length(m)
        zidx = findfirst(m_ -> m_ == 0, m)
        lpt(k) = k == zid ? log(d.θ+d.d*d.t) - z : m[k] == 0 ? -Inf : log(m[k]-d.d) - z
        return map(k -> lpt(k), 1:K)
    else
        z = log(sum(m) + d.θ)
        K = length(m)
        lp(k) = k > K ? log(d.θ + d.d*d.t) - z : log(m[k] - d.d) - z
        return map(k -> lp(k), 1:(K+1))
    end
end

## ####### ##
## Exports ##
## ####### ##

export DirichletProcess, PitmanYorProcess
export SizeBiasedSamplingProcess, StickBreakingProcess, ChineseRestaurantProcess


end # end module
