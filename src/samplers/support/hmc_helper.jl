#####################################
# Helper functions for Dual numbers #
#####################################

function realpart(d)
  if isa(d[1,1], Dual)      # matrix
    return map(x -> Float64(x.value), d)
  elseif isa(d[1,1], Array) # array of arry
    return [map(x -> Float64(x.value), d[i]) for i in 1:length(d)]
  end
end

function dualpart(d)
  return map(x -> Float64(x), d.partials.values)
end

function make_dual(dim, real, idx)
  z = zeros(dim)
  z[idx] = 1
  return Dual(real, tuple(collect(z)...))
end

Base.convert(::Type{Float64}, d::Dual{0,Float64}) = d.value
Base.convert(::Type{Float64}, d::Dual{0,Int64}) = round(Int, d.value)
Base.convert(::Type{Int64}, d::Dual{0,Int64}) = d.value

#####################################################
# Helper functions for vectorize/reconstruct values #
#####################################################

function vectorize(d::UnivariateDistribution, r)
  if isa(r, Dual)
    val = Vector{Any}([r])
  else
    val = Vector{Any}([Dual(r)])
  end
  val
end

function vectorize(d::MultivariateDistribution, r)
  if isa(r[1], Dual)
    val = Vector{Any}(map(x -> x, r))
  else
    val = Vector{Any}(map(x -> Dual(x), r))
  end
  val
end

function vectorize(d::MatrixDistribution, r)
  if isa(r[1,1], Dual)
    val = Vector{Any}(map(x -> x, vec(r)))
  else
    s = Dual(sum(r))
    val = Vector{Any}(map(x -> Dual(x), vec(r)))
  end
  val
end

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
