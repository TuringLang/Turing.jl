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

using StatsBase
# Effective Sample Size
function ess(samples)
  """
  @input:
    samples : a vector of samples
  @output:
    ess = n / (1 + 2∑ρ)
  """
  n = length(samples)
  acfs = StatsBase.autocor(samples, [1:(n - 1), deman=false])
  # Truncate the array according to http://search.r-project.org/library/LaplacesDemon/html/ESS.html
  acfs_trancated = []
  for i = 1:n
    if acfs[i] < 0.05
      break
    end
    push!(acfs_trancated, acfs[i])
  end
  # filter!(a -> a > 0.05, acfs)
  return n / (1 + 2 * sum(acfs_trancated))
end
