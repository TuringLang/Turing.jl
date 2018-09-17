#=
  NOTE: Codes below are adapted from
  https://github.com/brian-j-smith/Mamba.jl/blob/master/src/distributions/transformdistribution.jl
  The Mamba.jl package is licensed under the MIT License:
  > Copyright (c) 2014: Brian J Smith and other contributors:
  >
  > https://github.com/brian-j-smith/Mamba.jl/contributors
  >
  > Permission is hereby granted, free of charge, to any person obtaining
  > a copy of this software and associated documentation files (the
  > "Software"), to deal in the Software without restriction, including
  > without limitation the rights to use, copy, modify, merge, publish,
  > distribute, sublicense, and/or sell copies of the Software, and to
  > permit persons to whom the Software is furnished to do so, subject to
  > the following conditions:
  >
  > The above copyright notice and this permission notice shall be
  > included in all copies or substantial portions of the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#############
# a ≦ x ≦ b #
#############

const TransformDistribution{T<:ContinuousUnivariateDistribution} = Union{T, Truncated{T}}

function link(d::TransformDistribution, x::Real)
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return StatsFuns.logit((x - a) / (b - a))
    elseif lowerbounded
        return log(x - a)
    elseif upperbounded
        return log(b - x)
    else
        return x
    end
end

function invlink(d::TransformDistribution, y::Real)
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return (b - a) * StatsFuns.logistic(y) + a
    elseif lowerbounded
        return exp(y) + a
    elseif upperbounded
        return b - exp(y)
    else
        return y
    end
end

function logpdf_with_trans(d::TransformDistribution, x::Real, transform::Bool)
    lp = logpdf(d, x)
    if transform
        a, b = minimum(d), maximum(d)
        lowerbounded, upperbounded = isfinite(a), isfinite(b)
        if lowerbounded && upperbounded
            lp += log((x - a) * (b - x) / (b - a))
        elseif lowerbounded
            lp += log(x - a)
        elseif upperbounded
            lp += log(b - x)
        end
    end
    return lp
end


###############
# -∞ < x < -∞ #
###############

const RealDistribution = Union{
    Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist,
}

link(d::RealDistribution, x::Real) = x
invlink(d::RealDistribution, y::Real) = y
logpdf_with_trans(d::RealDistribution, y::Real, transform::Bool) = logpdf(d, y)


#########
# 0 < x #
#########

const PositiveDistribution = Union{
    BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma,
    InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull,
}

link(d::PositiveDistribution, x::Real) = log(x)
invlink(d::PositiveDistribution, y::Real) = exp(y)
function logpdf_with_trans(d::PositiveDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x)
end


#############
# 0 < x < 1 #
#############

const UnitDistribution = Union{Beta, KSOneSided, NoncentralBeta}

link(d::UnitDistribution, x::Real) = StatsFuns.logit(x)
invlink(d::UnitDistribution, y::Real) = StatsFuns.logistic(y)
function logpdf_with_trans(d::UnitDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x * (one(x) - x))
end


###########
# ∑xᵢ = 1 #
###########

const SimplexDistribution = Union{Dirichlet}

function link(d::SimplexDistribution, x::AbstractVector{T}) where T<:Real
    y, K = similar(x), length(x)

    sum_tmp = zero(T)
    z = x[1]
    y[1] = StatsFuns.logit(z) - log(one(T) / (K - 1))
    @inbounds for k in 2:K - 1
        sum_tmp += x[k - 1]
        z = x[k] / (one(T) - sum_tmp)
        y[k] = StatsFuns.logit(z) - log(one(T) / (K - k))
    end
    y[K] = zero(T)

    return y
end

# Vectorised implementation of the above.
function link(d::SimplexDistribution, X::AbstractMatrix{T}) where T<:Real
    Y, K, N = similar(X), size(X, 1), size(X, 2)

    @inbounds for n in 1:size(X, 2)
        sum_tmp, z = zero(T), X[1, n]
        Y[1, n] = StatsFuns.logit(z) - log(one(T) / (K - 1))
        for k in 2:K - 1
            sum_tmp += X[k - 1, n]
            z = X[k, n] / (one(T) - sum_tmp)
            Y[k, n] = StatsFuns.logit(z) - log(one(T) / (K - k))
        end
        Y[K, n] = zero(T)
    end

    return Y
end

function invlink(d::SimplexDistribution, y::AbstractVector{T}) where T<:Real
    x, K = similar(y), length(y)

    z = StatsFuns.logistic(y[1] + log(one(T) / (K - 1)))
    x[1] = z
    sum_tmp = zero(T)
    @inbounds for k = 2:K - 1
        z = StatsFuns.logistic(y[k] + log(one(T) / (K - k)))
        sum_tmp += x[k-1]
        x[k] = (one(T) - sum_tmp) * z
    end
    sum_tmp += x[K-1]
    x[K] = one(T) - sum_tmp

    return x
end

# Vectorised implementation of the above.
function invlink(d::SimplexDistribution, Y::AbstractMatrix{T}) where T<:Real
    X, K, N = similar(Y), size(Y, 1), size(Y, 2)

    @inbounds for n in 1:size(X, 2)
        sum_tmp, z = zero(T), StatsFuns.logistic(Y[1, n] + log(one(T) / (K - 1)))
        X[1, n] = z
        for k in 2:K - 1
            z = StatsFuns.logistic(Y[k, n] + log(one(T) / (K - k)))
            sum_tmp += X[k - 1]
            X[k, n] = (one(T) - sum_tmp) * z
        end
        sum_tmp += X[K - 1, n]
        X[K, n] = one(T) - sum_tmp
    end

    return X
end

function logpdf_with_trans(
    d::SimplexDistribution,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    lp = logpdf(d, x)
    if transform
        K = length(x)

        sum_tmp = zero(eltype(x))
        z = x[1]
        lp += log(z) + log(one(eltype(x)) - z)
        @inbounds for k in 2:K-1
            sum_tmp += x[k-1]
            z = x[k] / (one(eltype(x)) - sum_tmp)
            lp += log(z) + log(one(eltype(x)) - z) + log(one(eltype(x)) - sum_tmp)
        end
    end
    return lp
end

# REVIEW: why do we put this piece of code here?
function logpdf_with_trans(d::Categorical, x::Int)
    return d.p[x] > 0.0 && insupport(d, x) ? log(d.p[x]) : eltype(d.p)(-Inf)
end


###############
# MvLogNormal #
###############

using Distributions: AbstractMvLogNormal

link(d::AbstractMvLogNormal, x::AbstractVector{<:Real}) = log.(x)
invlink(d::AbstractMvLogNormal, y::AbstractVector{<:Real}) = exp.(y)
function logpdf_with_trans(
    d::AbstractMvLogNormal,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    return logpdf(d, x) + transform * sum(log, x)
end

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{InverseWishart, Wishart}

function link(d::PDMatDistribution, X::AbstractMatrix{T}) where {T<:Real}
    Y = cholesky(X).L
    for m in 1:size(Y, 1)
        Y[m, m] = log(Y[m, m])
    end
    return Matrix(Y)
end

function invlink(d::PDMatDistribution, Y::AbstractMatrix{T}) where {T<:Real}
    X, dim = copy(Y), size(Y)
    for m in 1:size(X, 1)
        X[m, m] = exp(X[m, m])
    end
    return LowerTriangular(X) * LowerTriangular(X)'
end

function logpdf_with_trans(d::PDMatDistribution, X::AbstractMatrix{<:Real}, transform::Bool)
    lp = logpdf(d, X)
    if transform && isfinite(lp)
        U = cholesky(X).U
        for i in 1:dim(d)
            lp += (dim(d) - i + 2) * log(U[i, i])
        end
        lp += dim(d) * log(2.0)
    end
    return lp
end


############################################
# Defaults (assume identity link function) #
############################################

# UnivariateDistributions
using Distributions: UnivariateDistribution

link(d::UnivariateDistribution, x::Real) = x
link(d::UnivariateDistribution, x::AbstractVector{<:Real}) = link.(Ref(d), x)

invlink(d::UnivariateDistribution, y::Real) = y
invlink(d::UnivariateDistribution, y::AbstractVector{<:Real}) = invlink.(Ref(d), y)

logpdf_with_trans(d::UnivariateDistribution, x::Real, ::Bool) = logpdf(d, x)
function logpdf_with_trans(
    d::UnivariateDistribution,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    return logpdf_with_trans.(Ref(d), x, transform)
end

# MultivariateDistributions
using Distributions: MultivariateDistribution

link(d::MultivariateDistribution, x::AbstractVector{<:Real}) = copy(x)
function link(d::MultivariateDistribution, X::AbstractMatrix{<:Real})
    Y = similar(X)
    for n in 1:size(X, 2)
        Y[:, n] = link(d, view(X, :, n))
    end
    return Y
end

invlink(d::MultivariateDistribution, y::AbstractVector{<:Real}) = copy(y)
function invlink(d::MultivariateDistribution, Y::AbstractMatrix{<:Real})
    X = similar(Y)
    for n in 1:size(Y, 2)
        X[:, n] = invlink(d, view(Y, :, n))
    end
    return X
end

function logpdf_with_trans(d::MultivariateDistribution, x::AbstractVector{<:Real}, ::Bool)
    return logpdf(d, x)
end
function logpdf_with_trans(
    d::MultivariateDistribution,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    return [logpdf_with_trans(d, view(X, :, n), transform) for n in 1:size(X, 2)]
end

# MatrixDistributions
using Distributions: MatrixDistribution

link(d::MatrixDistribution, X::AbstractMatrix{<:Real}) = copy(X)
link(d::MatrixDistribution, X::AbstractVector{<:AbstractMatrix{<:Real}}) = link.(Ref(d), X)

invlink(d::MatrixDistribution, Y::AbstractMatrix{<:Real}) = copy(Y)
function invlink(d::MatrixDistribution, Y::AbstractVector{<:AbstractMatrix{<:Real}})
    return invlink.(Ref(d), Y)
end

logpdf_with_trans(d::MatrixDistribution, X::AbstractMatrix{<:Real}, ::Bool) = logpdf(d, X)
function logpdf_with_trans(
    d::MatrixDistribution,
    X::AbstractVector{<:AbstractMatrix{<:Real}},
    transform::Bool,
)
    return logpdf_with_trans.(Ref(d), X, Ref(transform))
end
