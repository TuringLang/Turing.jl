#####################################
# Helper functions for Dual numbers #
#####################################

realpart(f)        = f
realpart(d::Dual)  = d.value
realpart(d::Array) = map(x -> x.value, d)
dualpart(d::Dual)  = d.partials.values
dualpart(d::Array) = map(x -> x.partials.values, d)

# (HG): Why do we need this function?
@suppress_err begin
  Base.promote_rule{N1,N2,A<:Real,B<:Real}(D1::Type{Dual{N1,A}}, D2::Type{Dual{N2,B}}) = Dual{max(N1, N2), promote_type(A, B)}
end

Base.promote_rule(D1::Type{Float64}, D2::Type{Dual}) = D2

Base.convert{N,T<:Real}(::Type{T}, d::Dual{N,T}) = d.value
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
    val = length(val) == 1 ? val[1] : val
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

function varInfo2samples(vi)
  samples = Dict{Symbol, Any}()
  for uid in keys(vi)
    val = vi[uid]
    if istrans(vi, uid)
      dist = getdist(vi, uid)
      val = reconstruct(dist, val)
      val = invlink(dist, val)
      val = Any[realpart(val)]
      val = length(val) == 1 ? val[1] : val   # Remove un-necessary []'s
    end
    samples[sym(uid)] = val
  end
  samples
end
