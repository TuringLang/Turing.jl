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
  chunk_size = 8
  if dim > 8
    min_prof_log = Inf
    n = ceil(Int, dim / 50)
    while (sz = ceil(Int, dim / n)) > 8
      println("[Turing] profiling chunk size = $(sz)")
      setchunksize(sz)
      prof_log = @elapsed for _ = 1:rep_num mf() end
      if prof_log < min_prof_log
        chunk_size = sz
        min_prof_log = prof_log
      end
      n += 1
    end
    println("[Turing] final chunk size chosen = $(chunk_size)")
  end
  setchunksize(chunk_size)
end
