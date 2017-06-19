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

vectorize{T<:Real}(d::UnivariateDistribution,   r::T)         = Vector{Real}([r])
vectorize{T<:Real}(d::MultivariateDistribution, r::Vector{T}) = Vector{Real}(r)
vectorize{T<:Real}(d::MatrixDistribution,       r::Matrix{T}) = Vector{Real}(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
reconstruct(d::Distribution, val::Vector) = reconstruct(d, val, typeof(val[1]))
reconstruct(d::UnivariateDistribution,   val::Vector, T::Type) = T(val[1])
reconstruct(d::MultivariateDistribution, val::Vector, T::Type) = Array{T, 1}(val)
reconstruct(d::MatrixDistribution,       val::Vector, T::Type) = Array{T, 2}(reshape(val, size(d)...))

reconstruct(d::Distribution, val::Vector, n::Int) = reconstruct(d, val, typeof(val[1]), n)
reconstruct(d::UnivariateDistribution,   val::Vector, T::Type, n::Int) = Array{T, 1}(val)
reconstruct(d::MultivariateDistribution, val::Vector, T::Type, n::Int) = Array{T, 2}(reshape(val, size(d)[1], n))
reconstruct(d::MatrixDistribution,       val::Vector, T::Type, n::Int) = begin
  orig = Vector{Matrix{T}}(n)
  tmp = Array{T, 3}(reshape(val, size(d)[1], size(d)[2], n))
  for i = 1:n
    orig[i] = tmp[:,:,i]
  end
  orig
end

##########
# Others #
##########

# VarInfo to Sample
Sample(vi::VarInfo) = begin
  value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
  for vn in keys(vi)
    value[sym(vn)] = realpart(vi[vn])
  end
  # NOTE: do we need to check if lp is 0?
  value[:lp] = realpart(getlogp(vi))

  Sample(0.0, value)
end

# VarInfo, combined with spl.info, to Sample
Sample(vi::VarInfo, spl::Sampler) = begin
  s = Sample(vi)

  if haskey(spl.info, :ϵ)
    s.value[:epsilon] = spl.info[:ϵ][end]
  end

  if haskey(spl.info, :lf_num)
    s.value[:lf_num] = spl.info[:lf_num][end]
  end

  s
end
