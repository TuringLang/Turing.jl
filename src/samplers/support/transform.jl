import Distributions.logpdf

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

function invlink(d::Dirichlet, y::Vector, logpdf=false)
  K = length(y)
  T = typeof(y[1])
  z = [loginvlogit(y[k] - log(K - k)) for k in 1:K-1]
  x = Vector{T}(K)
  for k in 1:K-1
    x[k] = log(-expm1(logsumexp(x[1:k-1]))) + z[k]
  end
  x[K] = log(-expm1(logsumexp(x[1:K-1])))
  if logpdf
    exp(x), logpdf(d, Nullable(x), true)
  else
    exp(x)
  end
end

function logpdf(d::SimplexDistribution, x::Union{Vector,Nullable}, transform::Bool)
  # if typeof(x) == Nullable{Vector}, then x is in log scale.
  ## Step 0.
  logx :: Vector = isa(x, Nullable)? get(x) : log(x)

  ## Step 1: Compute logpdf(d, x)
  # x is in the log scale
  a = d.alpha
  s = 0.
  for i in 1:length(a)
    # @inbounds s += (a[i] - 1.0) * log(x[i])
    @inbounds s += (a[i] - 1.0) * logx[i]
  end
  lp = s - d.lmnB
  ## Step 2: Compute the jocabian term if transform is true.
  if transform
    x = exp(logx)
    K = length(x)
    T = typeof(x[1])
    logz = Vector{T}(K-1)
    for k in 1:K-1
      # z[k] = x[k] / (1 - sum(x[1:k-1]))
      logz[k] = logx[k] - log(-expm1(logsumexp(logx[1:k-1])))
    end
    # lp += sum([log(z[k]) + log(1 - z[k]) + log(1 - sum(x[1:k-1])) for k in 1:K-1])
    lp += sum([logz[k] + log(-expm1(logz[k])) + log(-expm1(logsumexp(logx[1:k-1]))) for k in 1:K-1])
  end
  lp
end

# julia> logpdf(Dirichlet([1., 1., 1.]), exp([-1000., -1000., -1000.]), true)
# NaN
# julia> logpdf(Dirichlet([1., 1., 1.]), Nullable([-1000., -1000., -1000.]), true)
# -1999.30685281944
#
# julia> logpdf(Dirichlet([1., 1., 1.]), exp([-1., -2., -3.]), true)
# -3.006450206744678
# julia> logpdf_logx(Dirichlet([1., 1., 1.]), Nullable([-1., -2., -3.]), true)
# -3.006450206744678

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

function logpdf(d::Distribution, x, transform::Bool)
  logpdf(d, x)
end
