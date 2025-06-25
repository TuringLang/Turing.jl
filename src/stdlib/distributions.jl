"""
    Flat()

The *flat distribution* is the improper distribution of real numbers that has the improper
probability density function

```math
f(x) = 1.
```
"""
struct Flat <: ContinuousUnivariateDistribution end

Base.minimum(::Flat) = -Inf
Base.maximum(::Flat) = Inf

Base.rand(rng::Random.AbstractRNG, d::Flat) = rand(rng)
Distributions.logpdf(::Flat, x::Real) = zero(x)

# For vec support
Distributions.logpdf(::Flat, x::AbstractVector{<:Real}) = zero(x)
Distributions.loglikelihood(::Flat, x::AbstractVector{<:Real}) = zero(eltype(x))

"""
    FlatPos(l::Real)

The *positive flat distribution* with real-valued parameter `l` is the improper distribution
of real numbers that has the improper probability density function

```math
f(x) = \\begin{cases}
0 & \\text{if } x \\leq l, \\\\
1 & \\text{otherwise}.
\\end{cases}
```
"""
struct FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
end

Base.minimum(d::FlatPos) = d.l
Base.maximum(::FlatPos) = Inf

Base.rand(rng::Random.AbstractRNG, d::FlatPos) = rand(rng) + d.l
function Distributions.logpdf(d::FlatPos, x::Real)
    z = float(zero(x))
    return x <= d.l ? oftype(z, -Inf) : z
end
# For vec support
function Distributions.loglikelihood(d::FlatPos, x::AbstractVector{<:Real})
    lower = d.l
    T = float(eltype(x))
    return any(xi <= lower for xi in x) ? T(-Inf) : zero(T)
end

"""
    BinomialLogit(n, logitp)

The *Binomial distribution* with logit parameterization characterizes the number of
successes in a sequence of independent trials.

It has two parameters: `n`, the number of trials, and `logitp`, the logit of the probability
of success in an individual trial, with the distribution

```math
P(X = k) = {n \\choose k}{(\\text{logistic}(logitp))}^k (1 - \\text{logistic}(logitp))^{n-k}, \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

See also: [`Binomial`](@ref)
"""
struct BinomialLogit{T<:Real,S<:Real} <: DiscreteUnivariateDistribution
    n::Int
    logitp::T
    logconstant::S

    function BinomialLogit{T}(n::Int, logitp::T) where {T}
        n >= 0 || error("parameter `n` has to be non-negative")
        logconstant = -(log1p(n) + n * StatsFuns.log1pexp(logitp))
        return new{T,typeof(logconstant)}(n, logitp, logconstant)
    end
end

BinomialLogit(n::Int, logitp::Real) = BinomialLogit{typeof(logitp)}(n, logitp)

Base.minimum(::BinomialLogit) = 0
Base.maximum(d::BinomialLogit) = d.n

function Distributions.logpdf(d::BinomialLogit, k::Real)
    n, logitp, logconstant = d.n, d.logitp, d.logconstant
    _insupport = insupport(d, k)
    _k = _insupport ? round(Int, k) : 0
    result = logconstant + _k * logitp - SpecialFunctions.logbeta(n - _k + 1, _k + 1)
    return _insupport ? result : oftype(result, -Inf)
end

function Base.rand(rng::Random.AbstractRNG, d::BinomialLogit)
    return rand(rng, Binomial(d.n, logistic(d.logitp)))
end
Distributions.sampler(d::BinomialLogit) = sampler(Binomial(d.n, logistic(d.logitp)))

"""
    OrderedLogistic(η, c::AbstractVector)

The *ordered logistic distribution* with real-valued parameter `η` and cutpoints `c` has the
probability mass function

```math
P(X = k) = \\begin{cases}
    1 - \\text{logistic}(\\eta - c_1) & \\text{if } k = 1, \\\\
    \\text{logistic}(\\eta - c_{k-1}) - \\text{logistic}(\\eta - c_k) & \\text{if } 1 < k < K, \\\\
    \\text{logistic}(\\eta - c_{K-1}) & \\text{if } k = K,
\\end{cases}
```
where `K = length(c) + 1`.
"""
struct OrderedLogistic{T1,T2<:AbstractVector} <: DiscreteUnivariateDistribution
    η::T1
    cutpoints::T2

    function OrderedLogistic{T1,T2}(η::T1, cutpoints::T2) where {T1,T2}
        issorted(cutpoints) || error("cutpoints are not sorted")
        return new{typeof(η),typeof(cutpoints)}(η, cutpoints)
    end
end

function OrderedLogistic(η, cutpoints::AbstractVector)
    return OrderedLogistic{typeof(η),typeof(cutpoints)}(η, cutpoints)
end

Base.minimum(d::OrderedLogistic) = 1
Base.maximum(d::OrderedLogistic) = length(d.cutpoints) + 1

function Distributions.logpdf(d::OrderedLogistic, k::Real)
    η, cutpoints = d.η, d.cutpoints
    K = length(cutpoints) + 1

    _insupport = insupport(d, k)
    _k = _insupport ? round(Int, k) : 1
    logp = unsafe_logpdf_ordered_logistic(η, cutpoints, K, _k)

    return _insupport ? logp : oftype(logp, -Inf)
end

function Base.rand(rng::Random.AbstractRNG, d::OrderedLogistic)
    η, cutpoints = d.η, d.cutpoints
    K = length(cutpoints) + 1
    # evaluate probability mass function
    ps = map(1:K) do k
        exp(unsafe_logpdf_ordered_logistic(η, cutpoints, K, k))
    end
    k = rand(rng, Categorical(ps))
    return k
end
function Distributions.sampler(d::OrderedLogistic)
    η, cutpoints = d.η, d.cutpoints
    K = length(cutpoints) + 1
    # evaluate probability mass function
    ps = map(1:K) do k
        exp(unsafe_logpdf_ordered_logistic(η, cutpoints, K, k))
    end
    return sampler(Categorical(ps))
end

# unsafe version without bounds checking
function unsafe_logpdf_ordered_logistic(η, cutpoints, K, k::Int)
    @inbounds begin
        logp = if k == 1
            -StatsFuns.log1pexp(η - cutpoints[k])
        elseif k < K
            tmp = StatsFuns.log1pexp(cutpoints[k - 1] - η)
            -tmp + StatsFuns.log1mexp(tmp - StatsFuns.log1pexp(cutpoints[k] - η))
        else
            -StatsFuns.log1pexp(cutpoints[k - 1] - η)
        end
    end
    return logp
end

"""
    LogPoisson(logλ)

The *Poisson distribution* with logarithmic parameterization of the rate parameter
describes the number of independent events occurring within a unit time interval, given the
average rate of occurrence ``\\exp(\\log\\lambda)``.

The distribution has the probability mass function

```math
P(X = k) = \\frac{e^{k \\cdot \\log\\lambda}}{k!} e^{-e^{\\log\\lambda}}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```

See also: [`Poisson`](@ref)
"""
struct LogPoisson{T<:Real,S} <: DiscreteUnivariateDistribution
    logλ::T
    λ::S

    function LogPoisson{T}(logλ::T) where {T}
        λ = exp(logλ)
        return new{T,typeof(λ)}(logλ, λ)
    end
end

LogPoisson(logλ::Real) = LogPoisson{typeof(logλ)}(logλ)

Base.minimum(d::LogPoisson) = 0
Base.maximum(d::LogPoisson) = Inf

function Distributions.logpdf(d::LogPoisson, k::Real)
    _insupport = insupport(d, k)
    _k = _insupport ? round(Int, k) : 0
    logp = _k * d.logλ - d.λ - SpecialFunctions.loggamma(_k + 1)

    return _insupport ? logp : oftype(logp, -Inf)
end

Base.rand(rng::Random.AbstractRNG, d::LogPoisson) = rand(rng, Poisson(d.λ))
Distributions.sampler(d::LogPoisson) = sampler(Poisson(d.λ))
