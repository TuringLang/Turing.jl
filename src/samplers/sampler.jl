doc"""
    ParticleSampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `run` function that produces results of inference. TODO: specify the format of this output
3. Methods for modelling functions specified in this file. This is where actual inference happens.

Turing translates models to thunks that call the modelling functions below at specified points.
Their dispatch is based on the value of a global variable `sampler`.
To include a new inference algorithm implement the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
type ParticleSampler{T} <: Sampler{T}
  alg         ::  T
  particles   ::  ParticleContainer
  ParticleSampler(alg::T) = begin
    s = new()
    s.alg = alg
    s
  end
end

# Concrete algorithm implementations.
include("support/helper.jl")
include("support/resample.jl")
@suppress_err begin
  include("support/transform.jl")
end
include("hmc.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("gibbs.jl")

## Fallback functions
assume(spl, distr :: Distribution) = begin
  error("[assume]: unmanaged inference algorithm: $(typeof(spl))")
end

observe(spl, weight :: Float64) = begin
  error("[observe]: unmanaged inference algorithm: $(typeof(spl))")
end

## Default definitions for assume, observe, when sampler = nothing.
assume(spl :: Void, dist :: Distribution, vn :: VarName, vi :: VarInfo) = begin
  r = rand(vi, vn, dist)
  r
end

observe(spl :: Void, d :: Distribution, value, vi :: VarInfo) = begin
  vi.logjoint += logpdf(d, value)
end

assume(spl :: ParticleSampler, d :: Distribution, vn :: VarName, vi) = begin
  rand(current_trace(), vn, d)
end

observe(spl :: ParticleSampler, d :: Distribution, value, vi) = begin
  lp          = logpdf(d, value)
  vi.logjoint += lp
  produce(lp)
end
