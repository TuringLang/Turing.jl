#####################################
# Helper functions for Dual numbers #
#####################################

function realpart(d)
  if isa(d[1,1], Dual)      # matrix
    map(x -> Float64(x.value), d)
  elseif isa(d[1,1], Array) # array of arry
    [map(x -> Float64(x.value), d[i]) for i in 1:length(d)]
  end
end

dualpart(d) = map(x -> Float64(x), d.partials.values)

function make_dual(dim, real, idx)
  z = zeros(dim)
  z[idx] = 1
  Dual(real, tuple(collect(z)...))
end

import Base.promote_rule
Base.promote_rule{N1,N2,A<:Real,B<:Real}(D1::Type{Dual{N1,A}}, D2::Type{Dual{N2,B}}) = Dual{max(N1, N2), promote_type(A, B)}

Base.convert{N,T<:Real}(::Type{T}, d::Dual{N,T})  = d.value
Base.convert(::Type{Float64}, d::Dual{0,Int}) = round(Int, d.value)

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

vectorize(d::UnivariateDistribution, r)   = Vector{Dual}([r])
vectorize(d::MultivariateDistribution, r) = Vector{Dual}(r)
vectorize(d::MatrixDistribution, r)       = Vector{Dual}(vec(r))

function reconstruct(d::Distribution, val)
  if isa(d, UnivariateDistribution)
    # Turn Array{Any} to Any if necessary (this is due to randn())
    val = val[1]
  elseif isa(d, MultivariateDistribution)
    # Turn Vector{Any} to Vector{T} if necessary (this is due to an update in Distributions.jl)
    T = typeof(val[1])
    val = Vector{T}(val)
  elseif isa(d, MatrixDistribution)
    T = typeof(val[1])
    val = Array{T, 2}(reshape(val, size(d)...))
  end
  val
end

export realpart, dualpart, make_dual, vectorize, reconstruct
