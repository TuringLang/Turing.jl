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
@inline realpart(ds::TArray) = realpart(Array(ds))

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

##########
# Others #
##########

# VarInfo to Sample
@inline Sample(vi::VarInfo) = begin
  value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
  for vn in keys(vi)
    value[sym(vn)] = realpart(vi[vn])
  end

  # NOTE: do we need to check if lp is 0?
  value[:lp] = realpart(getlogp(vi))



  if ~isempty(vi.pred)
    for sym in keys(vi.pred)
      # if ~haskey(sample.value, sym)
        value[sym] = vi.pred[sym]
      # end
    end
    # TODO: check why 1. 2. cause errors
    # TODO: which one is faster?
    # 1. Using empty!
    # empty!(vi.pred)
    # 2. Reassign an enmtpy dict
    # vi.pred = Dict{Symbol,Any}()
    # 3. Do nothing?
  end

  Sample(0.0, value)
end

# VarInfo, combined with spl.info, to Sample
@inline Sample(vi::VarInfo, spl::Sampler) = begin
  s = Sample(vi)

  if haskey(spl.info, :ϵ)
    s.value[:epsilon] = spl.info[:ϵ][end]
  end

  if haskey(spl.info, :lf_num)
    s.value[:lf_num] = spl.info[:lf_num][end]
  end

  s
end
