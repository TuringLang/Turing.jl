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
    logit((x - a) / (b - a))
  elseif lowerbounded
    log(x - a)
  elseif upperbounded
    log(b - x)
  else
    x
  end
end

function invlink(d::TransformDistribution, x::Real)
  a, b = minimum(d), maximum(d)
  lowerbounded, upperbounded = isfinite(a), isfinite(b)
  if lowerbounded && upperbounded
    (b - a) * invlogit(x) + a
  elseif lowerbounded
    exp(x) + a
  elseif upperbounded
    b - exp(x)
  else
    x
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
  lp
end


###############
# -∞ < x < -∞ #
###############

const RealDistribution = Union{Cauchy, Gumbel, Laplace, Logistic,
                               NoncentralT, Normal, NormalCanon, TDist}

link(d::RealDistribution, x::Real) = x

invlink(d::RealDistribution, x::Real) = x

logpdf_with_trans(d::RealDistribution, x::Real, transform::Bool) = logpdf(d, x)


#########
# 0 < x #
#########

const PositiveDistribution = Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet,
                                   Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal,
                                   NoncentralChisq, NoncentralF, Rayleigh, Weibull}

link(d::PositiveDistribution, x::Real) = log(x)

invlink(d::PositiveDistribution, x::Real) = exp(x)

function logpdf_with_trans(d::PositiveDistribution, x::Real, transform::Bool)
  return logpdf(d, x) + transform * log(x)
end


#############
# 0 < x < 1 #
#############

const UnitDistribution = Union{Beta, KSOneSided, NoncentralBeta}

link(d::UnitDistribution, x::Real) = logit(x)

invlink(d::UnitDistribution, x::Real) = invlogit(x)

function logpdf_with_trans(d::UnitDistribution, x::Real, transform::Bool)
  return logpdf(d, x) + transform * log(x * (one(x) - x))
end


###########
# ∑xᵢ = 1 #
###########

const SimplexDistribution = Union{Dirichlet}

link(d::SimplexDistribution, x::AbstractVector{<:Real}) = link!(similar(x), d, x)
function link!(
  y::AbstractVector{<:Real},
  d::SimplexDistribution,
  x::AbstractVector{<:Real},
)
  K, T = length(x), eltype(x)

  sum_tmp = zero(T)
  z = x[1]
  y[1] = logit(z) - log(one(T) / (K-1))
  @simd for k in 2:K-1
    @inbounds sum_tmp += x[k-1]
    @inbounds z = x[k] / (one(T) - sum_tmp)
    @inbounds y[k] = logit(z) - log(one(T) / (K-k))
  end

  y
end

invlink(d::SimplexDistribution, y::AbstractVector{<:Real})= invlink!(similar(y), d, y)
function invlink!(
  x::AbstractVector{<:Real},
  d::SimplexDistribution,
  y::AbstractVector{<:Real},
)
  K, T = length(y), eltype(y)

  z = invlogit(y[1] + log(one(T) / (K - 1)))
  x[1] = z
  sum_tmp = zero(T)
  @simd for k = 2:K-1
    @inbounds z = invlogit(y[k] + log(one(T) / (K - k)))
    @inbounds sum_tmp += x[k-1]
    @inbounds x[k] = (one(T) - sum_tmp) * z
  end
  sum_tmp += x[K-1]
  x[K] = one(T) - sum_tmp

  x
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
    @simd for k in 2:K-1
      @inbounds sum_tmp += x[k-1]
      @inbounds z = x[k] / (one(eltype(x)) - sum_tmp)
      @inbounds lp += log(z) + log(one(eltype(x)) - z) + log(one(eltype(x)) - sum_tmp)
    end
  end
  lp
end

# REVIEW: why do we put this piece of code here?
function logpdf_with_trans(d::Categorical, x::Int)
  d.p[x] > 0.0 && insupport(d, x) ? log(d.p[x]) : eltype(d.p)(-Inf)
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
  return Y
end

function invlink(d::PDMatDistribution, Y::LowerTriangular{T}) where {T<:Real}
  X, dim = copy(Y), size(Y)
  for m in 1:size(X, 1)
    X[m, m] = exp(X[m, m])
  end
  return X * X'
end

function logpdf_with_trans(d::PDMatDistribution, X::AbstractMatrix{<:Real}, transform::Bool)
  lp = logpdf(d, X)
  if transform && isfinite(lp)
    U = cholesky(X).U
    for i in 1:dim(d)
      lp += (dim(d) - i + 2.0) * log(U[i, i])
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

invlink(d::UnivariateDistribution, x::Real) = x
invlink(d::UnivariateDistribution, x::AbstractVector{<:Real}) = invlink.(Ref(d), x)

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

link(d::MultivariateDistribution, x::AbstractVector{<:Real}) = link!(similar(x), d, x)
function link!(
  y::AbstractVector{<:Real},
  d::MultivariateDistribution,
  x::AbstractVector{<:Real},
)
  return copyto!(y, x)
end
link(d::MultivariateDistribution, X::AbstractMatrix{<:Real}) = link(similar(X), d, X)
function invlink!(
  Y::AbstractMatrix{<:Real},
  d::MultivariateDistribution,
  X::AbstractMatrix{<:Real},
)
  return [link!(view(Y, :, n), d, view(X, :, n)) for n in 1:size(X, 2)]
end

invlink(d::MultivariateDistribution, y::AbstractVector{<:Real}) = invlink!(similar(y), d, y)
function invlink!(
  x::AbstractVector{<:Real},
  d::MultivariateDistribution,
  y::AbstractVector{<:Real})
  return copyto!(x, y)
end
invlink(d::MultivariateDistribution, Y::AbstractMatrix{<:Real}) = invlink!(similar(Y), d, Y)
function invlink!(
  X::AbstractMatrix{<:Real},
  d::MultivariateDistribution,
  Y::AbstractMatrix{<:Real},
)
  return [invlink!(view(X, :, n), d, view(Y, :, n)) for n in 1:size(Y, 2)]
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

link(d::MatrixDistribution, X::AbstractMatrix{<:Real}) = X
link(d::MatrixDistribution, X::AbstractVector{<:AbstractMatrix{<:Real}}) = link.(Ref(d), X)

invlink(d::MatrixDistribution, Y::AbstractMatrix{<:Real}) = Y
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


# link(d::SimplexDistribution, X::AbstractMatrix{<:Real}) = link!(similar(X), d, X)
# function link!(Y, d::SimplexDistribution, X::AbstractMatrix{T}) where {T<:Real}
#   nrow, ncol = size(X)
#   K = nrow

#   key = (:cache_mat, T, nrow - 1, ncol)
#   if key in keys(TRANS_CACHE)
#     Z = TRANS_CACHE[key]
#   else
#     Z = Matrix{T}(undef, nrow - 1, ncol)
#     TRANS_CACHE[key] = Z
#   end

#   for k in 1:K-1
#     Z[k, :] = X[k, :] ./ (one(T) .- sum(X[1:k-1, :], dims=1))'
#   end

#   @simd for k in 1:K-1
#     @inbounds Y[k, :] = logit.(Z[k, :]) .- log.(one(T) ./ (K - k))
#   end

#   Y
# end

# invlink(d::SimplexDistribution, Y::AbstractMatrix{<:Real}) = invlink!(similar(Y), d, Y)
# function invlink!(X, d::SimplexDistribution, Y::AbstractMatrix{T}) where {T<:Real}
#   nrow, ncol = size(Y)
#   K = nrow

#   key = (:cache_mat, T, nrow - 1, ncol)
#   if key in keys(TRANS_CACHE)
#     Z = TRANS_CACHE[key]
#   else
#     Z = Matrix{T}(nrow - 1, ncol)
#     TRANS_CACHE[key] = Z
#   end

#   for k in 1:K-1
#     tmp = Y[k, :]
#     @inbounds Z[k, :] = invlogit.(tmp .+ log.(one(T) / (K - k)))
#   end

#   for k in 1:K-1
#     X[k, :] = (one(T) .- sum(X[1:k-1, :], dims=1)) .* Z[k, :]'
#   end
#   X[K, :] = one(T) .- sum(X[1:K-1, :], dims=1)

#   # X[1,:] = Z[1,:]'
#   # sum_tmp = 0
#   # for k = 2:K-1
#   #   sum_tmp += X[k-1,:]
#   #   X[k,:] = (one(T) - sum_tmp') .* Z[k,:]
#   # end
#   # sum_tmp += X[K-1,:]
#   # X[K,:] = one(T) - sum_tmp'

#   X
# end
