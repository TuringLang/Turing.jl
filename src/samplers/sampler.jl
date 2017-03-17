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
include("support/transform.jl")
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

# Default functions
function sample(model::Function, alg::InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(sampler)
end

function sample(model::Function, data::Dict, alg::InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(model, data, sampler)
end

function sample(model::Function, alg::InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(model, Dict(), sampler)
end

assume(spl::ParticleSampler, dist::Distribution, uid::String, sym::Symbol, vi)  = rand(current_trace(), dist)

function assume(spl::ParticleSampler{PG}, dist::Distribution, uid::String, sym::Symbol, vi::VarInfo)
  if spl == nothing || isempty(spl.alg.space) || sym in spl.alg.space
    vi.syms[uid] = sym  # record symbol
    vi.vals[uid] = nothing
    vi.dists[uid] = dist
    r = rand(current_trace(), dist)     # gen random
  else  # if it isn't in space
    if haskey(vi.vals, uid)
      val = vi[uid]
      dist = vi.dists[uid]
      val = reconstruct(dist, val)
      r = invlink(dist, val)
      produce(logpdf(dist, r, true))
    else
      vi.syms[uid] = sym  # record symbol
      r = rand(current_trace(), dist)   # gen random
      produce(log(1.0))
    end
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

function predict(spl::Sampler, vi::VarInfo, task)
  for sym in syms(vi)
    predict(spl, sym, get(task, sym))
  end
end
