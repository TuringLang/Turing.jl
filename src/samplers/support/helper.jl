#####################################
# Helper functions for Dual numbers #
#####################################

realpart(r::Real)   = r
realpart(d::ForwardDiff.Dual)       = d.value
realpart(ds::Union{Array,SubArray}) :: Vector{Float64} = map(d -> realpart(d), ds)  # NOTE: this function is assumed to return a Vector now; before it may also return an 2D Array
realpart!(arr::Union{Array,SubArray}, ds::Union{Array,SubArray}) :: Vector{Float64} = begin
  for i = 1:length(ds)
    arr[i] = realpart(ds[i])
  end
end

dualpart(d::ForwardDiff.Dual)       = d.partials.values
dualpart(ds::Union{Array,SubArray}) = map(d -> dualpart(d), ds)

Base.promote_rule(D1::Type{Real}, D2::Type{ForwardDiff.Dual}) = D2

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
reconstruct(d::Distribution, val::Union{Vector,SubArray}) = reconstruct(d, val, typeof(val[1]))
reconstruct(d::UnivariateDistribution,   val::Union{Vector,SubArray}, T::Type) = T(val[1])
reconstruct(d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type) = Array{T, 1}(val)
reconstruct(d::MatrixDistribution,       val::Union{Vector,SubArray}, T::Type) = Array{T, 2}(reshape(val, size(d)...))

reconstruct(d::Distribution, val::Union{Vector,SubArray}, n::Int) = reconstruct(d, val, typeof(val[1]), n)
reconstruct(d::UnivariateDistribution,   val::Union{Vector,SubArray}, T::Type, n::Int) = Array{T, 1}(val)
reconstruct(d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type, n::Int) = Array{T, 2}(reshape(val, size(d)[1], n))
reconstruct(d::MatrixDistribution,       val::Union{Vector,SubArray}, T::Type, n::Int) = begin
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
    # TODO: the function below doesn't call realpart(::Array) now because the signature is changed to use the fact that Vector is faster; can potentialy change this
    value[sym(vn)] = map(d -> realpart(d), vi[vn])
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
