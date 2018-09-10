#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d::UnivariateDistribution, r::Real) = Vector{Real}([r])
vectorize(d::MultivariateDistribution, r::AbstractVector{<:Real}) = Vector{Real}(r)
vectorize(d::MatrixDistribution, r::AbstractMatrix{<:Real}) = Vector{Real}(vec(r))

# NOTE:
# We cannot use reconstruct{T} because val is always Vector{Real} then T will be Real.
# However here we would like the result to be specifric type, e.g. Array{Dual{4,Float64}, 2},
# otherwise we will have error for MatrixDistribution.
# Note this is not the case for MultivariateDistribution so I guess this might be lack of
# support for some types related to matrices (like PDMat).
function reconstruct(d::Distribution, val::Union{Vector,SubArray})
    return reconstruct(d, val, typeof(val[1]))
end
# NOTE: the implementation below was: `T(val[1])`; it is changed to `val[1]` due to the need of ReverseDiff.jl, no side effect found yet
reconstruct(d::UnivariateDistribution, val::Union{Vector,SubArray}, T::Type) = val[1]
function reconstruct(d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type)
    return Vector{T}(val)
end
function reconstruct(d::MatrixDistribution, val::Union{Vector,SubArray}, T::Type)
    return Array{T, 2}(reshape(val, size(d)...))
end

function reconstruct!(r, d::Distribution, val::Union{Vector,SubArray})
    return reconstruct!(r, d, val, typeof(val[1]))
end
function reconstruct!(r, d::MultivariateDistribution, val::Union{Vector,SubArray}, T::Type)
    r .= val
    return r
end
function reconstruct(d::Distribution, val::Union{Vector,SubArray}, n::Int)
    return reconstruct(d, val, typeof(val[1]), n)
end
function reconstruct(
    d::UnivariateDistribution,
    val::Union{Vector,SubArray},
    T::Type,
    n::Int,
)
    return Vector{T}(val)
end
function reconstruct(
    d::MultivariateDistribution,
    val::Union{Vector,SubArray},
    T::Type,
    n::Int,
)
    return Matrix{T}(reshape(val, size(d)[1], n))
end
function reconstruct(d::MatrixDistribution, val::Union{Vector,SubArray}, T::Type, n::Int)
    orig = Vector{Matrix{T}}(undef, n)
    tmp = Array{T, 3}(reshape(val, size(d)[1], size(d)[2], n))
    for i = 1:n
        orig[i] = tmp[:, :, i]
    end
    return orig
end
function reconstruct!(r, d::Distribution, val::Union{Vector,SubArray}, n::Int)
    return reconstruct!(r, d, val, typeof(val[1]), n)
end
function reconstruct!(
    r,
    d::MultivariateDistribution,
    val::Union{Vector,SubArray},
    T::Type,
    n::Int,
)
    r .= val
    return r
end
