########
# Math #
########

@inline invlogit(x::Union{T,Vector{T},Matrix{T}}) where T<:Real = one(T) ./ (one(T) .+ exp.(-x))
@inline logit(x::Union{T,Vector{T},Matrix{T}}) where T<:Real = log.(x ./ (one(T) - x))
@init @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
  @inline invlogit(x::TrackedArray) = one(Real) ./ (one(Real) + exp.(-x))
  @inline logit(x::TrackedArray) = log.(x ./ (one(Real) - x))
end

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
end

align(x,y) = begin
  if length(x) < length(y)
    align_internal!(x, length(y))
  elseif length(x) > length(y)
    align_internal!(y, length(x))
  end

  (x,y)
end

#######
# I/O #
#######

macro sym_str(var)
  var_str = string(var)
  :(Symbol($var_str))
end

##########
# Helper #
##########

auto_tune_chunk_size!(mf::Function, rep_num=10) = begin
  dim = length(mf().vals)
  if dim > 8
    n = 1
    sz_cand = Int[]
    while dim / n > 8
      push!(sz_cand, ceil(Int, dim / n))
      n += 1
    end
    filter!(sz -> 8 < sz <= 50, sz_cand)
    sz_num = length(sz_cand)
    prof_log = Vector{Float64}(sz_num)
    for i = 1:sz_num
      println("[Turing] profiling chunk size = $(sz_cand[i])")
      setchunksize(sz_cand[i])
      prof_log[i] = @elapsed for _ = 1:rep_num mf() end
    end
    minval, minidx = findmin(prof_log)
    println("[Turing] final chunk size chosen = $(sz_cand[minidx])")
    setchunksize(sz_cand[minidx])
  else
    setchunksize(8)
  end
end
