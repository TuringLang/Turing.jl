# ---------   Utility Functions ----------- #
getvarid(s::Symbol) = string(":($s)")

getvarid(e::Expr)   = string("$(getvarid(e.args[1])), $(e.args[2])")

macro getvarid(e)
  # usage: @getvarid x[2][1+5], will return a tuple like (:x, 2, 6)
  return parse( getvarid(e) )
end

invlogit(x) = 1.0 ./ (exp(-x) + 1.0)

logit(x) = log(x ./ (1.0 - x))

function randcat(p::Vector{Float64}) # More stable, faster version of rand(Categorical)
  # if(any(p .< 0)) error("Negative probabilities not allowed"); end
  r, s = rand(), 1.0
  for j = 1:length(p)
    r -= p[j]
    if(r <= 0.0) s = j; break; end
  end
  return s
end

type NotImplementedException <: Exception end

# Numerically stable sum of values represented in log domain.
function logsum(xs :: Vector{Float64})
  largest = maximum(xs)
  ys = map(x -> exp(x - largest), xs)
  result = log(sum(ys)) + largest
  return result
end

using Distributions
# KL-divergence
function kl(p::Normal, q::Normal)
  return (log(q.σ / p.σ) + (p.σ^2 + (p.μ - q.μ)^2) / (2 * q.σ^2) - 0.5)
end

function align(x,y)
  if length(x) < length(y)
    z = zeros(y)
    z[1:length(x)] = x
    x = z
  elseif length(x) > length(y)
    z = zeros(x)
    z[1:length(y)] = y
    y = z
  end

  return (x,y)
end

# REVIEW: this functions is no where used
# function kl(p :: Categorical, q :: Categorical)
#   a,b = align(p.p, q.p)
#   return kl_divergence(a,b)
# end

#####################################
# Helper functions for Dual numbers #
#####################################

function realpart(d)
  return map(x -> Float64(x.value), d)
end

function dualpart(d)
  return map(x -> Float64(x), d.partials.values)
end

function make_dual(dim, real, idx)
  z = zeros(dim)
  z[idx] = 1
  return Dual(real, tuple(collect(z)...))
end

Base.convert(::Type{Float64}, d::Dual{0,Float64}) = d.value

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

function vectorize(d::UnivariateDistribution, r)
  if isa(r, Dual)
    val = Vector{Any}([r])
  else
    val = Vector{Any}([Dual(r)])
  end
  val
end

function vectorize(d::MultivariateDistribution, r)
  if isa(r[1], Dual)
    val = Vector{Any}(map(x -> x, r))
  else
    val = Vector{Any}(map(x -> Dual(x), r))
  end
  val
end

function vectorize(d::MatrixDistribution, r)
  if isa(r[1,1], Dual)
    val = Vector{Any}(map(x -> x, vec(r)))
  else
    s = Dual(sum(r))
    val = Vector{Any}(map(x -> Dual(x), vec(r)))
    # Dual(f) can lose precision of f.
    # The trick below is to prevent when s == 1 and lose this constrain (which is important for simplex)
    val[end] = s - sum(val[end-1])
  end
  val
end

function reconstruct(d::Distribution, val)
  if isa(d, UnivariateDistribution)
    # Turn Array{Any} to Any if necessary (this is due to randn())
    val = val[1]
  elseif isa(d, MultivariateDistribution)
    # Turn Vector{Any} to Vector{T} if necessary (this is due to an update in Distributions.jl)
    T = typeof(val[1])
    val = Vector{T}(val)
  elseif isa(d, MatrixDistribution)
    T = typeof(val[1])
    val = Array{T, 2}(reshape(val, size(d)...))
  end
  val
end

export kl, align, realpart, dualpart, make_dual, vectorize, reconstruct
