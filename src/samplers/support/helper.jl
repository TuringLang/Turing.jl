#####################################
# Helper functions for Dual numbers #
#####################################

realpart(r::Real)   = r
realpart(d::Dual)   = d.value
realpart(ds::Array) = map(d -> realpart(d), ds)

dualpart(d::Dual)   = d.partials.values
dualpart(ds::Array) = map(d -> dualpart(d), ds)

Base.promote_rule(D1::Type{Real}, D2::Type{Dual}) = D2

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d::UnivariateDistribution,   r) = Vector{Real}([r])
vectorize(d::MultivariateDistribution, r) = Vector{Real}(r)
vectorize(d::MatrixDistribution,       r) = Vector{Real}(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
reconstruct(d::Distribution, val::Vector) = reconstruct(d, val, typeof(val[1]))
reconstruct(d::UnivariateDistribution,   val::Vector, T::Type) = T(val[1])
reconstruct(d::MultivariateDistribution, val::Vector, T::Type) = Vector{T}(val)
reconstruct(d::MatrixDistribution,       val::Vector, T::Type) = Array{T, 2}(reshape(val, size(d)...))

##########
# Others #
##########

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
