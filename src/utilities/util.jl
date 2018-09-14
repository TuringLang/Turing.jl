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
