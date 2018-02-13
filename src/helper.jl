#####################################
# Helper functions for Dual numbers #
#####################################

@inline realpart(r::Real)             = r
@inline realpart(d::ForwardDiff.Dual) = d.value
@inline realpart(ds::Union{Vector,SubArray}) = Float64[realpart(d) for d in ds] # NOTE: the function below is assumed to return a Vector now
@inline realpart!(arr::Union{Array,SubArray}, ds::Union{Array,SubArray}) = for i = 1:length(ds) arr[i] = realpart(ds[i]) end
@inline realpart{T<:Real}(ds::Matrix{T}) = Float64[realpart(col) for col in ds]
@inline realpart(ds::Matrix{Any}) = [realpart(col) for col in ds]
@inline realpart(ds::Array)  = map(d -> realpart(d), ds)  # NOTE: this function is not optimized
# @inline realpart(ds::TArray) = realpart(Array(ds))    # TODO: is it disabled temporary
@inline realpart(ta::ReverseDiff.TrackedArray) = ta.value[1]

@inline dualpart(d::ForwardDiff.Dual)       = d.partials.values
@inline dualpart(ds::Union{Array,SubArray}) = map(d -> dualpart(d), ds)

# Base.promote_rule(D1::Type{Real}, D2::Type{ForwardDiff.Dual}) = D2

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

@inline vectorize{T<:Real}(d::UnivariateDistribution,   r::T)         = Vector{Real}([r])
@inline vectorize{T<:Real}(d::MultivariateDistribution, r::Vector{T}) = Vector{Real}(r)
@inline vectorize{T<:Real}(d::MatrixDistribution,       r::Matrix{T}) = Vector{Real}(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
@inline reconstruct(d::Distribution, val::Union{Vector,SubArray}) = reconstruct(d, val, typeof(val[1]))
@inline reconstruct(d::UnivariateDistribution,   val::Union{Vector,SubArray}, T::Type) = T(val[1])
@inline reconstruct(d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type) = Array{T, 1}(val)
@inline reconstruct(d::MatrixDistribution,       val::Union{Vector,SubArray}, T::Type) = Array{T, 2}(reshape(val, size(d)...))

@inline reconstruct(d::Distribution, val::Union{Vector,SubArray}, n::Int) = reconstruct(d, val, typeof(val[1]), n)
@inline reconstruct(d::UnivariateDistribution,   val::Union{Vector,SubArray}, T::Type, n::Int) = Array{T, 1}(val)
@inline reconstruct(d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type, n::Int) = Array{T, 2}(reshape(val, size(d)[1], n))

@inline reconstruct(d::MatrixDistribution,       val::Union{Vector,SubArray}, T::Type, n::Int) = begin
  orig = Vector{Matrix{T}}(n)
  tmp = Array{T, 3}(reshape(val, size(d)[1], size(d)[2], n))
  for i = 1:n
    orig[i] = tmp[:,:,i]
  end
  orig
end