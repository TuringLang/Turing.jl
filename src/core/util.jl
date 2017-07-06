# ---------   Utility Functions ----------- #

invlogit{T<:Real}(x::Union{T,Vector{T},Matrix{T}}) = one(T) ./ (one(T) + exp(-x))
logit{T<:Real}(x::Union{T,Vector{T},Matrix{T}}) = log(x ./ (one(T) - x))

# More stable, faster version of rand(Categorical)
function randcat(p::Vector{Float64})
  # if(any(p .< 0)) error("Negative probabilities not allowed"); end
  r, s = rand(), one(Int)
  for j = 1:length(p)
    r -= p[j]
    if(r <= 0.0) s = j; break; end
  end

  s
end

type NotImplementedException <: Exception end

# Numerically stable sum of values represented in log domain.
function logsum(xs::Vector{Float64})
  largest = maximum(xs)
  ys = map(x -> exp(x - largest), xs)

  log(sum(ys)) + largest
end

# KL-divergence
kl(p::Normal, q::Normal) = (log(q.σ / p.σ) + (p.σ^2 + (p.μ - q.μ)^2) / (2 * q.σ^2) - 0.5)

align(x,y) = begin
  if length(x) < length(y)
    z = zeros(y)
    z[1:length(x)] = x
    x = z
  elseif length(x) > length(y)
    z = zeros(x)
    z[1:length(y)] = y
    y = z
  end

  (x,y)
end

macro sym_str(var)
  var_str = string(var)
  :(Symbol($var_str))
end
