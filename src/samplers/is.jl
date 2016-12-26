
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
  n_samples :: Int
end

type ImportanceSampler{IS} <: Sampler{IS}
  alg :: IS
  model :: Function
  samples :: Array{Dict{Symbol,Any}}
  logweights :: Array{Float64}
  logevidence :: Float64
  predicts :: Dict{Symbol,Any}
  function ImportanceSampler(alg :: IS, model :: Function)
    samples = Array{Dict{Symbol,Any}}(alg.n_samples)
    for i = 1:alg.n_samples
      samples[i] = Dict{Symbol,Any}()
    end
    logweights = zeros(Float64, alg.n_samples)
    logevidence = 0
    predicts = Dict{Symbol,Any}()
    new(alg, model, samples, logweights, logevidence, predicts)
  end
end

function Base.run(spl :: Sampler{IS})
  n = spl.alg.n_samples
  for i = 1:n
    consume(Task(spl.model))
    spl.samples[i] = spl.predicts
    spl.logweights[i] = spl.logevidence
    spl.logevidence = 0
    spl.predicts = Dict{Symbol,Any}()
  end
  spl.logevidence = logsum(spl.logweights) - log(n)
  results = Dict{Symbol,Any}()
  results[:logevidence] = spl.logevidence
  results[:logweights] = spl.logweights
  results[:samples] = spl.samples
  return results
end

function assume(spl :: ImportanceSampler{IS}, d :: Distribution, p, varInfo::GradientInfo)
  return rand(d)
end

function observe(spl :: ImportanceSampler{IS}, d :: Distribution, value, varInfo::GradientInfo)
  spl.logevidence += logpdf(d, value)
end

function predict(spl :: ImportanceSampler{IS}, name :: Symbol, value) spl.predicts[name] = value
end

sample(model :: Function, alg :: IS) =
  (global sampler = ImportanceSampler{IS}(alg, model); run(sampler))
