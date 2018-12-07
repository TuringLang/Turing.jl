export DirichletProcess, PitmanYorProcess
export SizeBiasedSamplingProcess, StickBreakingProcess, ChineseRestaurantProcess

abstract type AbstractRandomProbabilityMeasure end

"""
    SizeBiasedSamplingProcess(rpm, surplus)

The *Size-Biased Sampling Process* for random probability measures `rpm` with a surplus mass of `surplus`.
"""
struct SizeBiasedSamplingProcess <: ContinuousUnivariateDistribution
    rpm::AbstractRandomProbabilityMeasure
    surplus::Float64
end

logpdf(d::SizeBiasedSamplingProcess, x::Real) = logpdf_stickbreaking(d.rpm, x/surplus)
rand(d::SizeBiasedSamplingProcess) = d.surplus*stickbreaking(d.rpm)
minimum(d::SizeBiasedSamplingProcess) = 0.0
maximum(d::SizeBiasedSamplingProcess) = d.surplus

"""
    StickBreakingProcess(rpm)

The *Stick-Breaking Process* for random probability measures `rpm`.
"""
struct StickBreakingProcess <: ContinuousUnivariateDistribution
    rpm::AbstractRandomProbabilityMeasure
end

logpdf(d::StickBreakingProcess, x::Real) = logpdf_stickbreaking(d.rpm, x)
rand(d::StickBreakingProcess) = stickbreaking(d.rpm)
minimum(d::StickBreakingProcess) = 0.0
maximum(d::StickBreakingProcess) = 1.0

"""
    ChineseRestaurantProcess(rpm, m)

The *Chinese Restaurant Process* for random probability measures `rpm` with counts `m`.
"""
struct ChineseRestaurantProcess <: DiscreteUnivariateDistribution
    rpm::AbstractRandomProbabilityMeasure
    m::Vector{Int}
end

function logpdf(d::ChineseRestaurantProcess, x::Int)
    if insupport(d, x)
        lp = crp(d.rpm, d.m)
        return lp[x] - logsumexp(lp)
    else
        return -Inf
    end
end

function rand(d::ChineseRestaurantProcess)
    lp = crp(d.rpm, d.m)
    p = exp.(lp)
    return rand(Categorical(p ./ sum(p)))
end

minimum(d::ChineseRestaurantProcess) = 1
maximum(d::ChineseRestaurantProcess) = length(d.m) + 1

#abstract type TotalMassDistribution <: ContinuousUnivariateDistribution end
#Distributions.minimum(d::TotalMassDistribution) = 0.0
#Distributions.maximum(d::TotalMassDistribution) = Inf

########################
# Priors on Partitions #
########################

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
struct DirichletProcess <: AbstractRandomProbabilityMeasure
    α::Float64
end

stickbreaking(d::DirichletProcess) = rand(Beta(1., d.α))
logpdf_stickbreaking(d::DirichletProcess, x::T) where T<:Real = logpdf(Beta(1., d.α), x)

function crp(d::DirichletProcess, m::Vector{Int})
    if length(m) == 0
        return [0.0]
    end
    z = log(sum(m) - 1 + d.α)
    K = length(m)
    lp(k) = k > K ? log(d.α) - z : log(m[k]) - z
    return map(k -> lp(k), 1:(K+1))
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
struct PitmanYorProcess <: AbstractRandomProbabilityMeasure
    d::Float64
    θ::Float64
    t::Int
end

stickbreaking(d::PitmanYorProcess) = rand(Beta(1. - d.d, d.θ + d.t*d.d))
logpdf_stickbreaking(d::PitmanYorProcess, x::Real) = logpdf(Beta(1-d.d, d.θ + d.t*d.d), x)

function crp(d::PitmanYorProcess, m::Vector{Int})
    if length(m) == 0
        return [0.0]
    end
    z = log(sum(m) + d.θ)
    K = length(m)
    lp(k) = k > K ? log(d.θ + d.d*d.t) - z : log(m[k] - d.d) - z
    return map(k -> lp(k), 1:(K+1))
end
