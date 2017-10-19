
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
  SMC(n::Int, resampler::Function, resampler_threshold::Float64, use_replay::Bool, space::Set, gid::Int) = new(n, resampler, resampler_threshold, use_replay, space, gid)
  function SMC(n_particles::Int, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_particles, resampleSystematic, 0.5, false, space, 0)
  end
  SMC(alg::SMC, new_gid::Int) = new(alg.n_particles, alg.resampler, alg.resampler_threshold, alg.use_replay, alg.space, new_gid)
end

Sampler(alg::SMC) = begin
  info = Dict{Symbol, Any}()
  info[:logevidence] = []
  Sampler(alg, info)
end

step(model::Function, spl::Sampler{SMC}, vi::VarInfo) = begin
    TraceType = spl.alg.use_replay ? TraceR : TraceC
    particles = ParticleContainer{TraceType}(model)
    vi.index = 0; vi.num_produce = 0;  # We need this line cause fork2 deepcopy `vi`.
    vi[getretain(vi, 0, spl)] = NULL
    push!(particles, spl.alg.n_particles, spl, vi)

    while consume(particles) != Val{:done}
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler,use_replay=spl.alg.use_replay)
      end
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.info[:logevidence], particles.logE)

    particles[indx].vi
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
      resample!(particles,spl.alg.resampler,use_replay=spl.alg.use_replay)
    end
  end
  w, samples = getsample(particles)
  res = Chain(w, samples)

end
