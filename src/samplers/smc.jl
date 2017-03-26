
doc"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Usage:

```julia
SMC(1000)
```

Example:

```julia
@model example begin
  ...
end

sample(example, SMC(1000))
```
"""
immutable SMC <: InferenceAlgorithm
  n_particles :: Int
  resampler :: Function
  resampler_threshold :: Float64
  use_replay :: Bool
  SMC(n) = new(n, resampleSystematic, 0.5, false)
  SMC(n, b::Bool) = new(n, resampleSystematic, 0.5, b)
end

## wrapper for smc: run the sampler, collect results.
function Base.run(model, data, spl::Sampler{SMC})

  TraceType = spl.alg.use_replay ? TraceR : TraceC
  spl.particles = ParticleContainer{TraceType}(spl.model)
  push!(spl.particles, spl.alg.n_particles, data, spl, VarInfo())

  while consume(spl.particles) != Val{:done}
    ess = effectiveSampleSize(spl.particles)
    if ess <= spl.alg.resampler_threshold * length(spl.particles)
      resample!(spl.particles,use_replay=spl.alg.use_replay)
    end
  end

  res = Chain(getsample(spl.particles)...)

end
