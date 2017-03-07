#####################################
# Helper functions for Dual numbers #
#####################################

realpart(d::Dual)  = d.value
realpart(d::Array) = map(x -> x.value, d)
dualpart(d::Dual)  = d.partials.values
dualpart(d::Array) = map(x -> x.partials.values, d)

function make_dual(dim, real, idx)
  z = zeros(dim)
  z[idx] = 1
  Dual(real, z...)
end

# (HG): Why do we need this function?
@suppress_err begin
  import Base.promote_rule
  Base.promote_rule{N1,N2,A<:Real,B<:Real}(D1::Type{Dual{N1,A}}, D2::Type{Dual{N2,B}}) = Dual{max(N1, N2), promote_type(A, B)}
end

Base.convert{N,T<:Real}(::Type{T}, d::Dual{N,T})  = d.value
Base.convert{N}(::Type{Int}, d::Dual{N,Float64}) = round(Int, d.value)
Base.convert{N}(::Type{Int}, d::Dual{N,Float32}) = round(Int, d.value)
Base.convert{N}(::Type{Int}, d::Dual{N,Float16}) = round(Int, d.value)
Base.convert{N}(::Type{Float64}, d::Dual{N,Int}) = float(d.value)
Base.convert{N}(::Type{Float32}, d::Dual{N,Int}) = float(d.value)
Base.convert{N}(::Type{Float16}, d::Dual{N,Int}) = float(d.value)

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
