export DeterminantalPointProcess

"""
    DeterminantalPointProcess(L::Symmetric{<:AbstractFloat})

A determinantal point process with kernel `L`.
The kernel `L` is expected to be symmetric and positive-definite.

See: https://en.wikipedia.org/wiki/Determinantal_point_process
"""
struct DeterminantalPointProcess{T<:AbstractFloat} <: DiscreteMultivariateDistribution
    L::Symmetric{T}
    Leig::Eigen
    D::Int
end

function DeterminantalPointProcess(L::Symmetric{T}) where {T<:AbstractFloat}
    Leig = eigen(L)
    D = size(L)[1]
    return DeterminantalPointProcess(L, Leig, D)
end

function Distributions._logpdf(d::DeterminantalPointProcess, x::AbstractVector{Int})
    z = findall(x .== 1)
    Lx_eigvals = eigvals(d.L[z,z])
    return sum(log.(Lx_eigvals)) - sum(log.(d.Leig.values .+ 1))
end

Distributions.length(d::DeterminantalPointProcess) = d.D
