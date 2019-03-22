"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), SMC(1000))
```
"""
mutable struct SMC{T, F} <: InferenceAlgorithm
    n_particles           ::  Int
    resampler             ::  F
    resampler_threshold   ::  Float64
    space                 ::  Set{T}
end
SMC(n) = SMC(n, resample_systematic, 0.5, Set())
function SMC(n_particles::Int, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    SMC(n_particles, resample_systematic, 0.5, _space)
end

function Sampler(alg::SMC, s::Selector)
    info = Dict{Symbol, Any}()
    info[:logevidence] = []
    return Sampler(alg, info, s)
end

function step(model, spl::Sampler{<:SMC}, vi::VarInfo)
    particles = ParticleContainer{Trace}(model)
    vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    push!(particles, spl.alg.n_particles, spl, vi)

    while consume(particles) != Val{:done}
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler)
      end
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.info[:logevidence], particles.logE)

    return particles[indx].vi, true
end

## wrapper for smc: run the sampler, collect results.
function sample(model::Model, alg::SMC)
    spl = Sampler(alg)

    particles = ParticleContainer{Trace}(model)
    push!(particles, spl.alg.n_particles, spl, VarInfo())

    while consume(particles) != Val{:done}
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler)
      end
    end
    w, samples = getsample(particles)
    res = Chain(log(w), samples)
    return res
end
