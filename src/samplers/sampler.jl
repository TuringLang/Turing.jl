abstract InferenceAlgorithm{P}
abstract Sampler{T<:InferenceAlgorithm}
abstract GradientSampler{T} <: Sampler{T}

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
  alg ::  T
  particles :: ParticleContainer
  model :: Function
  ParticleSampler(m :: Function, a :: T) = (s = new(); s.alg = a; s.model = m; s)
end

# Concrete algorithm implementations.
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

predict(spl::Void, var_name :: Symbol, value) = nothing

function sample(model::Function, data::Dict, alg::InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(model, data, sampler)
end


assume(spl::ParticleSampler, dist::Distribution, uid::String, sym::Symbol, vi)  = rand(current_trace(), dist)

function assume(spl::ParticleSampler{PG}, dist::Distribution, uid::String, sym::Symbol, vi::VarInfo)
  ct = current_trace()
  r = rand(ct, dist)
  if ~haskey(ct.vi, uid)
    val = vectorize(dist, link(dist, r))  # X -> R and vectorize
    addvar!(ct.vi, uid, val, sym, dist)
  else
    nothing   # is actually replaying
  end
  if spl == nothing || isempty(spl.alg.space) || sym in spl.alg.space
    nothing
  else
    produce(log(1.0))
  end
  r
end
observe(spl :: ParticleSampler, d :: Distribution, value, varInfo) = produce(logpdf(d, value))

function predict(spl :: Sampler, v_name :: Symbol, value)
  task = current_task()
  if ~haskey(task.storage, :turing_predicts)
    task.storage[:turing_predicts] = Dict{Symbol,Any}()
  end
  task.storage[:turing_predicts][v_name] = isa(value, Dual) ? realpart(value) : value
end
