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

typealias TransformDistribution{T<:ContinuousUnivariateDistribution}
          Union{T, Truncated{T}}

link{T<:Real}(d::TransformDistribution, x::Union{T,Vector{T}}) = begin
  a, b = minimum(d), maximum(d)
  lowerbounded, upperbounded = isfinite(a), isfinite(b)
  if lowerbounded && upperbounded
    logit((x - a) ./ (b - a))
  elseif lowerbounded
    log(x - a)
  elseif upperbounded
    log(b - x)
  else
    x
  end
end

invlink{T<:Real}(d::TransformDistribution, x::Union{T,Vector{T}}) = begin
  a, b = minimum(d), maximum(d)
  lowerbounded, upperbounded = isfinite(a), isfinite(b)
  if lowerbounded && upperbounded
    (b - a) .* invlogit(x) + a
  elseif lowerbounded
    exp(x) + a
  elseif upperbounded
    b - exp(x)
  else
    x
  end
end

Distributions.logpdf{T<:Real}(d::TransformDistribution, x::Union{T,Vector{T}}, transform::Bool) = begin
  lp = logpdf(d, x)
  if transform
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
      lp += log((x - a) .* (b - x) ./ (b - a))
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

typealias RealDistribution
          Union{Cauchy, Gumbel, Laplace, Logistic,
                NoncentralT, Normal, NormalCanon, TDist}

link{T<:Real}(d::RealDistribution, x::Union{T,Vector{T}}) = x

invlink{T<:Real}(d::RealDistribution, x::Union{T,Vector{T}}) = x

Distributions.logpdf{T<:Real}(d::RealDistribution, x::Union{T,Vector{T}}, transform::Bool) = logpdf(d, x)


#########
# 0 < x #
#########

typealias PositiveDistribution
          Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet,
                Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal,
                NoncentralChisq, NoncentralF, Rayleigh, Weibull}

link{T<:Real}(d::PositiveDistribution, x::Union{T,Vector{T}}) = log(x)

invlink{T<:Real}(d::PositiveDistribution, x::Union{T,Vector{T}}) = exp(x)

Distributions.logpdf{T<:Real}(d::PositiveDistribution, x::Union{T,Vector{T}}, transform::Bool) = begin
  lp = logpdf(d, x)
  transform ? lp + log(x) : lp
end


#############
# 0 < x < 1 #
#############

typealias UnitDistribution
          Union{Beta, KSOneSided, NoncentralBeta}

link{T<:Real}(d::UnitDistribution, x::Union{T,Vector{T}}) = logit(x)

invlink{T<:Real}(d::UnitDistribution, x::Union{T,Vector{T}}) = invlogit(x)

Distributions.logpdf{T<:Real}(d::UnitDistribution, x::Union{T,Vector{T}}, transform::Bool) = begin
  lp = logpdf(d, x)
  transform ? lp + log(x .* (one(x) - x)) : lp
end

###########
# ∑xᵢ = 1 #
###########

typealias SimplexDistribution Union{Dirichlet}

link{T}(d::SimplexDistribution, x::Vector{T}) = begin
  K = length(x)
  z = Vector{T}(K-1)
  for k in 1:K-1
    z[k] = x[k] / (one(T) - sum(x[1:k-1]))
  end
  y = [logit(z[k]) - log(one(T) / (K-k)) for k in 1:K-1]
  push!(y, zero(T))
end

invlink{T}(d::SimplexDistribution, y::Vector{T}) = begin
  K = length(y)
  z = [invlogit(y[k] + log(one(T) / (K - k))) for k in 1:K-1]
  x = Vector{T}(K)
  for k in 1:K-1
    x[k] = (one(T) - sum(x[1:k-1])) * z[k]
  end
  x[K] = one(T) - sum(x[1:K-1])
  x
end

Distributions.logpdf{T}(d::SimplexDistribution, x::Vector{T}, transform::Bool) = begin
  lp = logpdf(d, x)
  if transform
    K = length(x)
    z = Vector{T}(K-1)
    for k in 1:K-1
      z[k] = x[k] / (one(T) - sum(x[1:k-1]))
    end
    lp += sum([log(z[k]) + log(one(T) - z[k]) + log(one(T) - sum(x[1:k-1])) for k in 1:K-1])
  end
  lp
end

Distributions.logpdf(d::Categorical, x::Int) = begin
  d.p[x] > 0.0 && insupport(d, x) ? log(d.p[x]) : eltype(d.p)(-Inf)
end

#####################
# Positive definite #
#####################

typealias PDMatDistribution Union{InverseWishart, Wishart}

link{T}(d::PDMatDistribution, x::Array{T,2}) = begin
  z = chol(x)'
  dim = size(z)
  for m in 1:dim[1]
    z[m, m] = log(z[m, m])
  end
  for m in 1:dim[1], n in m+1:dim[2]
    z[m, n] = zero(T)
  end
  Array{T,2}(z)
end

invlink{T}(d::PDMatDistribution, z::Array{T,2}) = begin
  dim = size(z)
  for m in 1:dim[1]
    z[m, m] = exp(z[m, m])
  end
  for m in 1:dim[1], n in m+1:dim[2]
    z[m, n] = zero(T)
  end
  Array{T,2}(z * z')
end

Distributions.logpdf{T}(d::PDMatDistribution, x::Array{T,2}, transform::Bool) = begin
  lp = logpdf(d, x)
  if transform && isfinite(lp)
    U = chol(x)
    n = dim(d)
    for i in 1:n
      lp += (n - i + T(2)) * log(U[i, i])
    end
    lp += n * log(T(2))
  end
  lp
end

#############
# Callbacks #
#############

link(d::Distribution, x::Any) = x

invlink(d::Distribution, x::Any) = x

Distributions.logpdf(d::Distribution, x::Any, transform::Bool) = logpdf(d, x)
