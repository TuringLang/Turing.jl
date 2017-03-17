# ---------   Utility Functions ----------- #
macro VarName(ex::Union{Expr, Symbol})
  if isa(ex, Symbol)
    _ = string(ex)
    return :(Symbol($_))
  elseif ex.head == :ref
    _2 = ex
    _1 = ""
    while _2.head == :ref
      if length(_2.args) > 2
        _1 = string([_2.args[2:end]...]) * ", $_1"
      else
        _1 = string(_2.args[2]) * ", $_1"
      end
      _2   = _2.args[1]
      isa(_2, Symbol) && (_1 = ":($_2)" * ", $_1"; break)
    end
    return parse(_1)
  else
    error("VarName: Mis-formed variable name $(e)!")
  end
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

export kl, align
