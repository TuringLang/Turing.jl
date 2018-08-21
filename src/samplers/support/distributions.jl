# No info
struct Flat <: ContinuousUnivariateDistribution
end

Distributions.rand(d::Flat) = rand()
Distributions.logpdf(d::Flat, x::T) where {T<:Real} = zero(x)
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf

# For vec support
Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf(d::Flat, x::Vector{T}) where {T<:Real} = zero(x)


# Pos
struct FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
    FlatPos{T}(l::T) where {T} = new{T}(l)
end

FlatPos(l::T) where {T<:Real} = FlatPos{T}(l)

Distributions.rand(d::FlatPos) = rand() + d.l
Distributions.logpdf(d::FlatPos, x::T) where {T<:Real} = if x <= d.l -Inf else zero(x) end
Distributions.minimum(d::FlatPos) = d.l
Distributions.maximum(d::FlatPos) = +Inf

# For vec support
Distributions.rand(d::FlatPos, n::Int) = Vector([rand() for _ = 1:n] .+ d.l)
Distributions.logpdf(d::FlatPos, x::Vector{T}) where {T<:Real} = if any(x .<= d.l) -Inf else zero(x) end
