import Random: AbstractRNG

# No info
struct Flat <: ContinuousUnivariateDistribution end

Distributions.rand(rng::AbstractRNG, d::Flat) = rand(rng)
Distributions.logpdf(d::Flat, x::Real) = zero(x)
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf

# For vec support
Distributions.logpdf(d::Flat, x::AbstractVector{<:Real}) = zero(x)

# Pos
struct FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
end

Distributions.rand(rng::AbstractRNG, d::FlatPos) = rand(rng) + d.l
Distributions.logpdf(d::FlatPos, x::Real) = x <= d.l ? -Inf : zero(x)
Distributions.minimum(d::FlatPos) = d.l
Distributions.maximum(d::FlatPos) = Inf

# For vec support
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

struct OrderedLogistic{T1, T2} <: DiscreteUnivariateDistribution
   η::T1
   cutpoints::Vector{T2}
end

function Distributions.logpdf(d::OrderedLogistic, k::Int)

    K = length(d.cutpoints)+1

    c =  d.cutpoints

    if k==1
        logp= - softplus(-(c[k]-d.η))  #logp= log(logistic(c[k]-d.η))
    elseif k<K
        logp= log(logistic(c[k]-d.η) - logistic(c[k-1]-d.η))
    else
        logp= - softplus(c[k-1]-d.η)  #logp= log(1-logistic(c[k-1]-d.η))
    end

    return logp
end

function Distributions.rand(rng::AbstractRNG, d::OrderedLogistic)
    cutpoints = d.cutpoints
    η = d.η

    K = length(cutpoints)+1
    c = vcat(-Inf, cutpoints, Inf)

    ps = [logistic(η - i[1]) - logistic(η - i[2]) for i in zip(c[1:(end-1)],c[2:end])]

    k = rand(rng, Categorical(ps))

    if all(ps.>0)
        return(k)
    else
        return(-Inf)
    end
end
