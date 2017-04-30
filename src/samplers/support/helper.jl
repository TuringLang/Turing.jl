#####################################
# Helper functions for Dual numbers #
#####################################

realpart(f)        = f
realpart(d::Dual)  = d.value
realpart(d::Array) = map(x -> realpart(x), d)
dualpart(d::Dual)  = d.partials.values
dualpart(d::Array) = map(x -> x.partials.values, d)

Base.promote_rule(D1::Type{Real}, D2::Type{Dual}) = D2

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

# QUES: will use Any below lead to worse performance?
vectorize(d::UnivariateDistribution,   r) = Vector{Real}([r])
vectorize(d::MultivariateDistribution, r) = Vector{Real}(r)
vectorize(d::MatrixDistribution,       r) = Vector{Real}(vec(r))

function reconstruct(d::Distribution, val)
  if isa(d, UnivariateDistribution)
    # Turn Array{Any} to Any if necessary (this is due to randn())
    val = length(val) == 1 ? val[1] : val
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

export realpart, dualpart, make_dual, vectorize, reconstruct

# VarInfo to Sample
Sample(vi::VarInfo) = begin
  weight = 0.0
  value = Dict{Symbol, Any}()
  for uid in keys(vi)
    dist = getdist(vi, uid)
    r = reconstruct(dist, vi[uid])
    r = istransformed(vi, uid) ? invlink(dist, r) : r
    value[sym(uid)] = realpart(r)
  end
  # NOTE: do we need to check if lp is 0?
  value[:lp] = realpart(vi.logp)
  Sample(weight, value)
end

function cleandual!(vi::VarInfo)
  for uid in keys(vi)
    range = getrange(vi, uid)
    vi[range] = realpart(vi[uid])
  end
  vi.logp = realpart(vi.logp)
  vi.logw = realpart(vi.logw)
end

# X -> R for all variables associated with given sampler
function link(_vi, spl)
  vi = deepcopy(_vi)
  gkeys = spl == nothing ?
          keys(vi) :
          groupuids(vi, spl.alg.group_id, spl)
  for k in gkeys
    dist = getdist(vi, k)
    vi[k] = vectorize(dist, link(dist, reconstruct(dist, vi[k])))
    settrans!(vi, true, k)
  end
  vi
end

# R -> X for all variables associated with given sampler
function invlink(_vi, spl)
  vi = deepcopy(_vi)
  gkeys = spl == nothing ?
          keys(vi) :
          groupuids(vi, spl.alg.group_id, spl)
  for k in gkeys
    dist = getdist(vi, k)
    vi[k] = vectorize(dist, invlink(dist, reconstruct(dist, vi[k])))
    settrans!(vi, false, k)
  end
  vi
end
