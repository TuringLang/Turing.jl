
doc"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Usage:

```julia
SMC(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), SMC(1000))
```
"""
immutable SMC <: InferenceAlgorithm
  n_particles           ::  Int
  resampler             ::  Function
  resampler_threshold   ::  Float64
  use_replay            ::  Bool
  space                 ::  Set
  gid                   ::  Int
  SMC(n) = new(n, resampleSystematic, 0.5, false, Set(), 0)
  SMC(n, b::Bool) = new(n, resampleSystematic, 0.5, b, Set(), 0)
end

Sampler(alg::SMC) = begin
  info = Dict{Symbol, Any}()
  # For explicit return
  info[:pred] = Dict{Symbol,Any}()
  Sampler(alg, info)
end

## wrapper for smc: run the sampler, collect results.
function sample(model::Function, alg::SMC)
  spl = Sampler(alg);

  TraceType = spl.alg.use_replay ? TraceR : TraceC
  particles = ParticleContainer{TraceType}(model)
  push!(particles, spl.alg.n_particles, spl, VarInfo())

  while consume(particles) != Val{:done}
    ess = effectiveSampleSize(particles)
    if ess <= spl.alg.resampler_threshold * length(particles)
      resample!(particles,use_replay=spl.alg.use_replay)
    end
  end
  w, samples = getsample(particles)
  for s = samples
    update_pred(s, spl)
  end
  res = Chain(w, samples)

end
