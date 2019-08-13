# No info
struct Flat <: ContinuousUnivariateDistribution end

Distributions.rand(d::Flat) = rand()
Distributions.logpdf(d::Flat, x::T) where T<:Real= zero(x)
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf

# For vec support
Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf(d::Flat, x::AbstractVector{<:Real}) = zero(x)

# Pos
struct FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
end

Distributions.rand(d::FlatPos) = rand() + d.l
Distributions.logpdf(d::FlatPos, x::Real) = x <= d.l ? -Inf : zero(x)
Distributions.minimum(d::FlatPos) = d.l
Distributions.maximum(d::FlatPos) = Inf

# For vec support
Distributions.rand(d::FlatPos, n::Int) = Vector([rand() for _ = 1:n] .+ d.l)
function Distributions.logpdf(d::FlatPos, x::AbstractVector{<:Real})
    return any(x .<= d.l) ? -Inf : zero(x)
end

# Binomial with logit
struct BinomialLogit{T<:Real, I<:Integer} <: DiscreteUnivariateDistribution
    n::I
    logitp::T
end

struct VecBinomialLogit{T<:Real, I<:Integer} <: DiscreteUnivariateDistribution
    n::Vector{I}
    logitp::Vector{T}
end

function logpdf_binomial_logit(n, logitp, k)
    logcomb = -StatsFuns.log1p(n) - SpecialFunctions.lbeta(n - k + 1, k + 1)
    return logcomb + k * logitp - n * StatsFuns.log1pexp(logitp)
end

function Distributions.logpdf(d::BinomialLogit{<:Real}, k::Int)
    return logpdf_binomial_logit(d.n, d.logitp, k)
end

function Distributions.logpdf(d::VecBinomialLogit{<:Real}, ks::Vector{<:Integer})
    return sum(logpdf_binomial_logit.(d.n, d.logitp, ks))
end
