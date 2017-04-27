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


#################### TransformDistribution ####################

typealias TransformDistribution{T<:ContinuousUnivariateDistribution}
  Union{T, Truncated{T}}

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

Distributions.logpdf(d::TransformDistribution, x::Real, transform::Bool) = begin
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


#################### RealDistribution ####################

typealias RealDistribution
          Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal,
                NormalCanon, TDist}

link(d::RealDistribution, x::Real) = x

invlink(d::RealDistribution, x::Real) = x

Distributions.logpdf(d::RealDistribution, x::Real, transform::Bool) = logpdf(d, x)


#################### PositiveDistribution ####################

typealias PositiveDistribution
          Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet,
                Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal,
                NoncentralChisq, NoncentralF, Rayleigh, Weibull}

link(d::PositiveDistribution, x::Real) = log(x)

invlink(d::PositiveDistribution, x::Real) = exp(x)

Distributions.logpdf(d::PositiveDistribution, x::Real, transform::Bool) = begin
  lp = logpdf(d, x)
  transform ? lp + log(x) : lp
end


#################### UnitDistribution ####################

typealias UnitDistribution
          Union{Beta, KSOneSided, NoncentralBeta}

link(d::UnitDistribution, x::Real) = logit(x)

invlink(d::UnitDistribution, x::Real) = invlogit(x)

Distributions.logpdf(d::UnitDistribution, x::Real, transform::Bool) = begin
  lp = logpdf(d, x)
  transform ? lp + log(x * (1.0 - x)) : lp
end

################### SimplexDistribution ###################

typealias SimplexDistribution Union{Dirichlet}

function link(d::SimplexDistribution, x::Vector)
  K = length(x)
  T = typeof(x[1])
  z = Vector{T}(K-1)
  for k in 1:K-1
    z[k] = x[k] / (1 - sum(x[1:k-1]))
  end
  y = [logit(z[k]) - log(1 / (K-k)) for k in 1:K-1]
  push!(y, T(0))
end

function invlink(d::SimplexDistribution, y::Vector)
  K = length(y)
  T = typeof(y[1])
  z = [invlogit(y[k] + log(1 / (K - k))) for k in 1:K-1]
  x = Vector{T}(K)
  for k in 1:K-1
    x[k] = (1 - sum(x[1:k-1])) * z[k]
  end
  x[K] = 1 - sum(x[1:K-1])
  x
end

Distributions.logpdf(d::SimplexDistribution, x::Vector, transform::Bool) = begin
  lp = logpdf(d, x)
  if transform
    K = length(x)
    T = typeof(x[1])
    z = Vector{T}(K-1)
    for k in 1:K-1
      z[k] = x[k] / (1 - sum(x[1:k-1]))
    end
    lp += sum([log(z[k]) + log(1 - z[k]) + log(1 - sum(x[1:k-1])) for k in 1:K-1])
  end
  lp
end

Distributions.logpdf(d::Categorical, x::Int) = begin
  d.p[x] > 0.0 && insupport(d, x) ? log(d.p[x]) : eltype(d.p)(-Inf)
end

############### PDMatDistribution ##############

typealias PDMatDistribution Union{InverseWishart, Wishart}

function link(d::PDMatDistribution, x::Array)
  z = chol(x)'
  dim = size(z)
  for m in 1:dim[1]
    z[m, m] = log(z[m, m])
  end
  for m in 1:dim[1], n in m+1:dim[2]
    z[m, n] = 0
  end
  z
end

function invlink(d::PDMatDistribution, z::Union{Array, LowerTriangular})
  dim = size(z)
  for m in 1:dim[1]
    z[m, m] = exp(z[m, m])
  end
  for m in 1:dim[1], n in m+1:dim[2]
    z[m, n] = 0
  end
  z * z'
end

Distributions.logpdf(d::PDMatDistribution, x::Array, transform::Bool) = begin
  lp = logpdf(d, x)
  if transform && isfinite(lp)
    U = chol(x)
    n = dim(d)
    for i in 1:n
      lp += (n - i + 2) * log(U[i, i])
    end
    lp += n * log(2)
  end
  lp
end

################## Callback functions ##################

link(d::Distribution, x) = x

invlink(d::Distribution, x) = x

Distributions.logpdf(d::Distribution, x, transform::Bool) = logpdf(d, x)
