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

# Fallback functions
Base.run(spl :: Sampler) = error("[sample]: unmanaged inference algorithm: $(typeof(spl))")

assume(spl, distr :: Distribution) =
  error("[assume]: unmanaged inference algorithm: $(typeof(spl))")

observe(spl, weight :: Float64) =
  error("[observe]: unmanaged inference algorithm: $(typeof(spl))")

predict(spl, var_name :: Symbol, value) =
  error("[predict]: unmanaged inference algorithm: $(typeof(spl))")

function assume(spl::Void, dist::Distribution, vn::VarName, vi::VarInfo)
  r = rand(vi, vn, dist)
  trans = istransformed(vi, vn)
  vi.logjoint += logpdf(dist, r, trans)
  r
end

predict(spl::Void, var_name :: Symbol, value) = nothing

rand(vi::VarInfo, vn::VarName, dist::Distribution, spl:: Void) = begin
  # NOTE: Void sampler uses replaying by name method by default
  rand(vi, vn, dist, :byname)
end

function sample(model::Function, data::Dict, alg::InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(model, data, sampler)
end

assume(spl::ParticleSampler, dist::Distribution, vn::VarName, vi)  = rand(current_trace(), vn, dist)

observe(spl :: ParticleSampler, d :: Distribution, value, varInfo) = produce(logpdf(d, value))

predict(spl :: Sampler, v_name :: Symbol, value) = nothing
