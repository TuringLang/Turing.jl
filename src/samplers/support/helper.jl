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
    length(val) == 1 ? val[1] : val
  elseif isa(d, MultivariateDistribution)
    # Turn Vector{Any} to Vector{T} if necessary (this is due to an update in Distributions.jl)
    T = typeof(val[1])
    Vector{T}(val)
  elseif isa(d, MatrixDistribution)
    T = typeof(val[1])
    Array{T, 2}(reshape(val, size(d)...))
  end
end

# VarInfo to Sample
Sample(vi::VarInfo) = begin
  weight = 0.0
  value = Dict{Symbol, Any}()
  for vn in keys(vi)
    dist = getdist(vi, vn)
    r = vi[vn]
    value[sym(vn)] = realpart(r)
  end
  # NOTE: do we need to check if lp is 0?
  value[:lp] = realpart(vi.logp)
  Sample(weight, value)
end
