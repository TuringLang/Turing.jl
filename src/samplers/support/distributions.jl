# No info
immutable Flat <: ContinuousUnivariateDistribution
end

Distributions.rand(d::Flat) = rand()
Distributions.logpdf{T<:Real}(d::Flat, x::T) = zero(x)
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf

# For vec support
Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf{T<:Real}(d::Flat, x::Vector{T}) = zero(x)


# Pos
immutable FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
    (::Type{FlatPos{T}}){T}(l::T) = new{T}(l)
end

FlatPos{T<:Real}(l::T) = FlatPos{T}(l)

Distributions.rand(d::FlatPos) = rand() + d.l
Distributions.logpdf{T<:Real}(d::FlatPos, x::T) = if x <= d.l -Inf else zero(x) end
Distributions.minimum(d::FlatPos) = d.l
Distributions.maximum(d::FlatPos) = +Inf

# For vec support
Distributions.rand(d::FlatPos, n::Int) = Vector([rand() for _ = 1:n] .+ d.l)
Distributions.logpdf{T<:Real}(d::FlatPos, x::Vector{T}) = if any(x .<= d.l) -Inf else zero(x) end
