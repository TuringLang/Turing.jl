# ---------   Utility Functions ----------- #
getvarid(s::Symbol) = string(":($s)")

getvarid(e::Expr)   = string("$(getvarid(e.args[1])), $(e.args[2])")

macro getvarid(e)
  # usage: @getvarid x[2][1+5], will return a tuple like (:x, 2, 6)
  return parse( getvarid(e) )
end

dot(x) = dot(x, x)

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

# Force a Vector to a single type vector Vector{T}
function forceVector(in_v, T :: DataType)
  l = length(in_v)
  out_v = zeros(T, l)
  for i = 1:l
    out_v[i] = T(in_v[i])
  end
  return out_v
end

function normalize!(x)
  norm = sum(x)
  x /= norm
  return x
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

function kl(p :: Categorical, q :: Categorical)
  a,b = align(p.p, q.p)
  return kl_divergence(a,b)
end


function realpart(d)
  return map(x -> Float64(x.value), d)
end


function dualpart(d)
  return map(x -> Float64(x.partials.values[1]), d)
end

export normalize!, kl, align, realpart, dualpart
