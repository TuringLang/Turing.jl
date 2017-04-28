# doc"""
#     Sampler{T}
#
# Generic interface for implementing inference algorithms.
# An implementation of an algorithm should include the following:
#
# 1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
# 2. A method of `run` function that produces results of inference. TODO: specify the format of this output
# 3. Methods for modelling functions specified in this file. This is where actual inference happens.
#
# Turing translates models to thunks that call the modelling functions below at specified points.
# Their dispatch is based on the value of a global variable `sampler`.
# To include a new inference algorithm implement the requirements mentioned above in a separate file,
# then include that file at the end of this one.
# """

# Concrete algorithm implementations.
include("support/helper.jl")
include("support/resample.jl")
@suppress_err begin
  include("support/transform.jl")
end
include("support/hmc_core.jl")
include("hmcda.jl")
include("nuts.jl")
include("hmc.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("gibbs.jl")

## Fallback functions

# utility funcs for querying sampler information
require_gradient(s :: Sampler) = false
require_particles(s :: Sampler) = false

assume(spl, distr :: Distribution) = begin
  error("[assume]: unmanaged inference algorithm: $(typeof(spl))")
end

observe(spl, weight :: Float64) = begin
  error("[observe]: unmanaged inference algorithm: $(typeof(spl))")
end

## Default definitions for assume, observe, when sampler = nothing.
assume(spl :: Void, dist :: Distribution, vn :: VarName, vi :: VarInfo) = begin
  r = rand(vi, vn, dist)
  # The following code has been merged into rand.
  # vi.logjoint += logpdf(dist, r, istransformed(vi, vn))
  r
end

observe(spl :: Void, d :: Distribution, value, vi :: VarInfo) = begin
  lp = logpdf(d, value)
  vi.logw     += lp
  vi.logjoint += lp
end

assume{T<:Union{PG,SMC}}(spl :: Sampler{T}, d :: Distribution, vn :: VarName, vi) = begin
  rand(current_trace(), vn, d)
end

observe{T<:Union{PG,SMC}}(spl :: Sampler{T}, d :: Distribution, value, vi) = begin
  lp          = logpdf(d, value)
  vi.logjoint += lp
  produce(lp)
end
