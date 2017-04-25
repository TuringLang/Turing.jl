import Distributions.logpdf

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
    #_ = logsumexp(log(b - a) + loginvlogit(x), log(a))
  elseif lowerbounded
    exp(x) + a
    # _ = logsumexp(x, log(a))
  elseif upperbounded
    b - exp(x) # b(1-exp(x)/b)
    # _ = log(b) + log1mexp(x-log(b)) #??
  else
    x
  end
end

function logpdf(d::TransformDistribution, x::Real, transform::Bool)
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

logpdf(d::RealDistribution, x::Real, transform::Bool) = logpdf(d, x)


#################### PositiveDistribution ####################

typealias PositiveDistribution
          Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet,
                Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal,
                NoncentralChisq, NoncentralF, Rayleigh, Weibull}

link(d::PositiveDistribution, x::Real) = log(x)

invlink(d::PositiveDistribution, x::Real) = exp(x)

function  logpdf(d::PositiveDistribution, x::Real, transform::Bool)
  lp = logpdf(d, x)
  transform ? lp + log(x) : lp
end


#################### UnitDistribution ####################

typealias UnitDistribution
          Union{Beta, KSOneSided, NoncentralBeta}

link(d::UnitDistribution, x::Real) = logit(x)

invlink(d::UnitDistribution, x::Real) = invlogit(x)

function logpdf(d::UnitDistribution, x::Real, transform::Bool)
  lp = logpdf(d, x)
  transform ? lp + log(x * (1.0 - x)) : lp
end

################### SimplexDistribution ###################

typealias SimplexDistribution Union{Dirichlet}

function link(d::SimplexDistribution, x::Vector)
  try @assert abs(sum(exp(x)) - 1) < 1e4
  catch e
    println(realpart(x))
    throw(e)
  end
  K = length(x)
  T = typeof(x[1])
  z = Vector{T}(K-1)
  for k in 1:K-1
    # z[k] = x[k] / (1 - sum(x[1:k-1]))
    z[k] = x[k] / (1 - sum(x[1:k-1]) + 1e-80) # Add small value for numerical stability.
  end
  y = [logit(z[k]) - log(1 / (K-k)) for k in 1:K-1]
  push!(y, T(0))
end

function Turing.invlink(d::Turing.SimplexDistribution, y::Vector, is_logx=false)
  K = length(y)
  T = typeof(y[1])
  # z = exp([loginvlogit(y[k] - log(K - k)) for k in 1:K-1])
  z = [invlogit(y[k] + log(1 / (K - k))) for k in 1:K-1]
  x = Vector{T}(K)
  for k in 1:K-1
    # x[k] = log(-expm1(logsumexp(x[1:k-1]))) + z[k]
    # try x[k] = log1mexp(logsumexp(x[1:k-1])) + z[k]
    #  0 <= exp(logsumexp(x[1:k-1])) <= 1
    try x[k] = (1-sum(x[1:k-1]))*exp(z[k])
    catch e
      println("y=$(realpart(y))")
      println("x=$(realpart(x))")
      println("k=$k")
      println("d=$d")
      throw(e)
    end
    # x[k] = logsumexp(0, logsumexp(-x[1:k-1])) + z[k]
  end
  # try x[K] = log1mexp(logsumexp(x[1:K-1]))
  try x[K] = 1 - sum(x[1:K-1])
  catch e
    println("y=$(realpart(y))")
    println("x=$(realpart(x))")
    throw(e)
  end
  #x[K] = logsumexp([0, -x[1:K-1]...])
  try @assert isprobvec(x)
  catch e
    println("y=$(realpart(y))")
    println("x=$(realpart(x))")
    throw(e)
  end
  # is_logx ? x : exp(x)
  is_logx ? log(x) : x
end

function logpdf(d::SimplexDistribution, x::Vector, transform::Bool)
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

# function Turing.logpdf(d::Turing.SimplexDistribution, x::Vector, transform::Bool, is_logx=false)
#   # NOTE: logx = log(x)
#   logx :: Vector = is_logx ? x : log(x)
#
#   ## Step 1: Compute logpdf(d, x)
#   # x is in the log scale
#   a = d.alpha
#   s = 0.
#   for i in 1:length(a)
#     # @inbounds s += (a[i] - 1.0) * log(x[i])
#     @inbounds s += (a[i] - 1.0) * logx[i]
#   end
#   lp = s - d.lmnB
#   ## Step 2: Compute the jocabian term if transform is true.
#   if transform
#     x = exp(logx)
#     K = length(x)
#     T = typeof(x[1])
#     logz = Vector{T}(K-1)
#     for k in 1:K-1
#       # z[k] = x[k] / (1 - sum(x[1:k-1]))
#       logz[k] = logx[k] - log1mexp(logsumexp(logx[1:k-1]))
#     end
#     # lp += sum([log(z[k]) + log(1 - z[k]) + log(1 - sum(x[1:k-1])) for k in 1:K-1])
#     # lp += sum([logz[k] + log1mexp(logz[k]) + log1mexp(logsumexp(logx[1:k-1])) for k in 1:K-1])
#     for k in 1:K-1
#       lp += logz[k] + log1mexp(logz[k]) + log1mexp(logsumexp(logx[1:k-1]))
#       if lp == -Inf
#         println(lp, k, log1mexp(logz[k]), log1mexp(logsumexp(logx[1:k-1])))
#       end
#     end
#   end
#   lp
# end

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

function logpdf(d::PDMatDistribution, x::Array, transform::Bool)
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

logpdf(d::Distribution, x, transform::Bool) = logpdf(d, x)
