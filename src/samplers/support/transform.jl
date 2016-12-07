import Distributions.logpdf

# ddylogit⁻¹(y) = logit⁻¹(y) * (1 - logit⁻¹(y)) # derivative of inverse log-odds
#
# function make_trans(a, b)
#   x -> logit((x - a) / (b - a))
# end
#
# function make_invtrans(a, b)
#   y -> a + (b - a) * logit⁻¹(y)
# end
#
# function make_py(px, invtrans, a, b)
#   y -> px(invtrans(y)) * (b - a) * ddylogit⁻¹(y)
# end
#
# trans = make_trans(0, 1)
# invtrans = make_invtrans(0, 1)
#
# dist = Beta(1, 2)
# px = x -> pdf(dist, x)
# py = make_py(px, invtrans, 0, 1)
#
# x = 0.3
# px(x)
#
# y = trans(x)
# py(y)
#

# NOTE: Codes below are adapted from https://github.com/brian-j-smith/Mamba.jl/blob/master/src/distributions/transformdistribution.jl

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

#################### Callback function ####################

link(d::Distribution, x) = x

invlink(d::Distribution, x) = x

function logpdf(d::Distribution, x::Real, transform::Bool)
  logpdf(d, x)
end
