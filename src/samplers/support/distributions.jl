# Temporary patch for GMM distribution to make AD work.

immutable UnivariateGMM2{T<:Real} <: UnivariateMixture{Continuous,Normal}
    K::Int
    means::Vector{T}
    stds::Vector{T}
    prior::Categorical

    function UnivariateGMM2(ms::Vector{T}, ss::Vector{T}, pri::Categorical)
        K = length(ms)
        length(ss) == K || throw(DimensionMismatch())
        ncategories(pri) == K ||
            error("The number of categories in pri should be equal to the number of components.")
        new(K, ms, ss, pri)
    end
end

UnivariateGMM2(ms::Vector{Float64}, ss::Vector{Float64}, pri::Categorical) = UnivariateGMM2{Float64}(ms, ss, pri)

Base.minimum(::Union{UnivariateGMM2,Type{UnivariateGMM2}}) = -Inf
Base.maximum(::Union{UnivariateGMM2,Type{UnivariateGMM2}}) = Inf

Distributions.ncomponents(d::UnivariateGMM2) = d.K

Distributions.component(d::UnivariateGMM2, k::Int) = Normal(d.means[k], d.stds[k])

Distributions.probs(d::UnivariateGMM2) = probs(d.prior)

Distributions.mean(d::UnivariateGMM2) = dot(d.means, probs(d))

Base.Random.rand(d::UnivariateGMM2) = (k = rand(d.prior); d.means[k] + randn() * d.stds[k])

Distributions.params(d::UnivariateGMM2) = (d.means, d.stds, d.prior)

immutable UnivariateGMM2Sampler <: Sampleable{Univariate,Continuous}
    means::Vector{Real}
    stds::Vector{Real}
    psampler::Distributions.AliasTable
end

Base.Random.rand(s::UnivariateGMM2Sampler) = (k = rand(s.psampler); s.means[k] + randn() * s.stds[k])
Distributions.sampler(d::UnivariateGMM2) = UnivariateGMM2Sampler(d.means, d.stds, sampler(d.prior))


function Distributions._mixpdf!(r::AbstractArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    fill!(r, 0.0)
    t = Array(typeof(x), size(r))
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            pdf!(t, component(d, i), x)
            BLAS.axpy!(pi, t, r)
        end
    end
    return r
end

function Distributions._mixlogpdf1(d::AbstractMixtureModel, x)
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))
    #              = m + log(sum_i exp(logpri[i] + logpdf(cs[i], x) - m))
    #
    #  m is chosen to be the maximum of logpri[i] + logpdf(cs[i], x)
    #  such that the argument of exp is in a reasonable range
    #

    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K

    lp = Array(typeof(x), K)
    m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            # lp[i] <- log(p(cs[i], x)) + log(pri[i])
            lp_i = logpdf(component(d, i), x) + log(pi)
            @inbounds lp[i] = lp_i
            if lp_i > m
                m = lp_i
            end
        end
    end
    v = 0.0
    @inbounds for i = 1:K
        if p[i] > 0.0
            v += exp(lp[i] - m)
        end
    end
    return m + log(v)
end

function Distributions._mixlogpdf!(r::AbstractArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    n = length(r)
    Lp = Array(typeof(x), n, K)
    m = fill(-Inf, n)
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            lpri = log(pi)
            lp_i = view(Lp, :, i)
            # compute logpdf in batch and store
            logpdf!(lp_i, component(d, i), x)

            # in the mean time, add log(prior) to lp and
            # update the maximum for each sample
            for j = 1:n
                lp_i[j] += lpri
                if lp_i[j] > m[j]
                    m[j] = lp_i[j]
                end
            end
        end
    end

    fill!(r, 0.0)
    @inbounds for i = 1:K
        if p[i] > 0.0
            lp_i = view(Lp, :, i)
            for j = 1:n
                r[j] += exp(lp_i[j] - m[j])
            end
        end
    end

    @inbounds for j = 1:n
        r[j] = log(r[j]) + m[j]
    end
    return r
end

# Temporary patch for trancated distributions.
#  Remove when PR #586 is accpted
#  https://github.com/JuliaStats/Distributions.jl/pull/586
Distributions.Truncated(d::UnivariateDistribution, l::Real, u::Real) = Distributions.Truncated(d, Float64(l), Float64(u))

Distributions.insupport{D<:UnivariateDistribution}(d::Distributions.Truncated{D,Union{Discrete,Continuous}}, x::Real) =
    d.lower <= x <= d.upper && insupport(d.untruncated, x)

Distributions.pdf{T<:Real}(d::Distributions.Truncated, x::T) = d.lower <= x <= d.upper ? pdf(d.untruncated, x) / d.tp : zero(T)

Distributions.logpdf{T<:Real}(d::Distributions.Truncated, x::T) = d.lower <= x <= d.upper ? logpdf(d.untruncated, x) - d.logtp : -T(Inf)

Distributions.cdf{T<:Real}(d::Distributions.Truncated, x::T) =
    x <= d.lower ? zero(T) :
    x >= d.upper ? one(T) :
    (cdf(d.untruncated, x) - d.lcdf) / d.tp

Distributions.logcdf{T<:Real}(d::Distributions.Truncated, x::T) =
    x <= d.lower ? -T(Inf) :
    x >= d.upper ? zero(T) :
    log(cdf(d.untruncated, x) - d.lcdf) - d.logtp

Distributions.ccdf{T<:Real}(d::Distributions.Truncated, x::T) =
    x <= d.lower ? one(T) :
    x >= d.upper ? zero(T) :
    (d.ucdf - cdf(d.untruncated, x)) / d.tp

Distributions.logccdf{T<:Real}(d::Distributions.Truncated, x::T) =
    x <= d.lower ? zero(T) :
    x >= d.upper ? -T(Inf) :
    log(d.ucdf - cdf(d.untruncated, x)) - d.logtp





using Distributions

# No info
immutable NoInfo <: ContinuousUnivariateDistribution
end

Distributions.rand(d::NoInfo) = rand()
Distributions.logpdf{T<:Real}(d::NoInfo, x::T) = zero(x)
Distributions.minimum(d::NoInfo) = -Inf
Distributions.maximum(d::NoInfo) = +Inf

# For vec support
Distributions.rand(d::NoInfo, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf{T<:Real}(d::NoInfo, x::Vector{T}) = zero(x)


# Pos
immutable NoInfoPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
    (::Type{NoInfoPos{T}}){T}(l::T) = new{T}(l)
end

NoInfoPos{T<:Real}(l::T) = NoInfoPos{T}(l)

Distributions.rand(d::NoInfoPos) = rand() + d.l
Distributions.logpdf{T<:Real}(d::NoInfoPos, x::T) = if x <= d.l -Inf else zero(x) end
Distributions.minimum(d::NoInfoPos) = d.l
Distributions.maximum(d::NoInfoPos) = +Inf

# For vec support
Distributions.rand(d::NoInfoPos, n::Int) = Vector([rand() for _ = 1:n] .+ d.l)
Distributions.logpdf{T<:Real}(d::NoInfoPos, x::Vector{T}) = if any(x .<= d.l) -Inf else zero(x) end
