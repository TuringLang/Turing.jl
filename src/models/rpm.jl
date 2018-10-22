export DirichletProcess, PitmanYorProcess

abstract type StickSizeBiasedDistribution <: ContinuousUnivariateDistribution end
abstract type SizeBiasedDistribution <: ContinuousUnivariateDistribution end
abstract type TotalMassDistribution <: ContinuousUnivariateDistribution end

Distributions.minimum(d::StickSizeBiasedDistribution) = 0.0
Distributions.maximum(d::StickSizeBiasedDistribution) = 1.0
init(dist::StickSizeBiasedDistribution) = rand(dist)

Distributions.minimum(d::SizeBiasedDistribution) = 0.0
Distributions.maximum(d::SizeBiasedDistribution) = d.t_surplus

Distributions.minimum(d::TotalMassDistribution) = 0.0
Distributions.maximum(d::TotalMassDistribution) = Inf

init(dist::SizeBiasedDistribution) = rand(dist)

########################
# Priors on Partitions #
########################

# Dirichlet Process
struct DirichletProcess <: SizeBiasedDistribution
    α::Float64
    surplus::Float64
end

Distributions.rand(d::DirichletProcess) = d.surplus*rand(Beta(1., d.α))
function Distributions.logpdf(d::DirichletProcess, x::T) where {T<:Real}
    return logpdf(Beta(1., d.α), x / d.surplus)
end

# Pitman-Yor Process
struct PitmanYorProcess <: SizeBiasedDistribution
    α::Float64
    d::Float64
    index::Int
    surplus::Float64
end

function Distributions.rand(d::PitmanYorProcess)
    return d.surplus*rand(Beta(1. - d.d, d.α + d.index*d.d))
end

function Distributions.logpdf(d::PitmanYorProcess, x::T) where {T<:Real}
    return logpdf(Beta(1. - d.d, d.α + d.index*d.d), x/d.surplus)
end

#######################################
# Normalised Inverse Gaussian Process #
#######################################

struct NormalizedInverseGP_T <: TotalMassDistribution
    τ::Float64
end

Distributions.rand(d::NormalizedInverseGP_T) = rand(ExpTiltedSigma(0.5, d.τ))
Distributions.logpdf(d::NormalizedInverseGP_T, x::T) where T<:Real = 0.
