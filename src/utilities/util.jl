########
# Math #
########

@inline invlogit(x::T) where T<:Real = one(T) / (one(T) + exp(-x))
@inline logit(x::T) where T<:Real = log(x / (one(T) - x))

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

struct NotImplementedException <: Exception end

# Numerically stable sum of values represented in log domain.
logsum(xs::Vector{T}) where T<:Real = begin
  largest = maximum(xs)
  ys = map(x -> exp.(x - largest), xs)

  log(sum(ys)) + largest
end

# KL-divergence
kl(p::Normal, q::Normal) = (log(q.σ / p.σ) + (p.σ^2 + (p.μ - q.μ)^2) / (2 * q.σ^2) - 0.5)

align_internal!(x,n) = begin
  m = length(x)
  resize!(x, n)
  x[m+1:end] .= zero(eltype(x))
  x
end

align(x,y) = begin
  if length(x) < length(y)
    align_internal!(x, length(y))
  elseif length(x) > length(y)
    align_internal!(y, length(x))
  end

  (x,y)
end
