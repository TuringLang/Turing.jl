
doc"""
    IS(n_particles::Int)

Importance sampler.

Usage:

```julia
IS(1000)
```

Example:

```julia
@model example begin
  ...
end

sample(example, IS(1000))
```
"""
immutable IS <: InferenceAlgorithm
  n_samples   ::  Int
end

type ImportanceSampler{IS} <: Sampler{IS}
  alg         ::  IS
  model       ::  Function
  samples     ::  Vector{Sample}
  logweights  ::  Array{Float64}
  logevidence ::  Float64
  predicts    ::  Dict{Symbol,Any}
  function ImportanceSampler(alg::IS, model::Function)
    samples = Array{Sample}(alg.n_samples)
    logweights = zeros(Float64, alg.n_samples)
    logevidence = 0
    predicts = Dict{Symbol,Any}()
    new(alg, model, samples, logweights, logevidence, predicts)
  end
end

function Base.run(spl::Sampler{IS})
  n = spl.alg.n_samples
  for i = 1:n
    consume(Task(spl.model))
    spl.samples[i] = Sample(spl.logevidence, spl.predicts)
    spl.logweights[i] = spl.logevidence
    spl.logevidence = 0
    spl.predicts = Dict{Symbol,Any}()
  end
  spl.logevidence = logsum(spl.logweights) - log(n)
  chn = Chain(exp(spl.logevidence), spl.samples)
  return chn
end

function assume(spl::ImportanceSampler{IS}, d::Distribution, uid::String, sym::Symbol, varInfo::VarInfo)
  return rand(d)
end

function observe(spl::ImportanceSampler{IS}, d::Distribution, value, varInfo::VarInfo)
  spl.logevidence += logpdf(d, value)
end

function predict(spl::ImportanceSampler{IS}, name::Symbol, value) spl.predicts[name] = value
end

sample(model::Function, alg::IS) =
  (global sampler = ImportanceSampler{IS}(alg, model); run(sampler))
