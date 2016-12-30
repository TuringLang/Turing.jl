# Particle Gibbs sampler

doc"""
    PG(n_particles::Int, n_iterations::Int)

Particle Gibbs sampler.

Usage:

```julia
PG(100, 100)
```

Example:

```julia
@model example begin
  ...
end

sample(example, PG(100, 100))
```
"""
immutable PG <: InferenceAlgorithm
  n_particles :: Int
  n_iterations :: Int
  resampler :: Function
  resampler_threshold :: Float64
  space :: Set
  PG(n1::Int, n2::Int) = new(n1, n2, resampleSystematic, 0.5, Set())
  function PG(n1::Int, n2::Int, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n1, n2, resampleSystematic, 0.5, space)
  end
end

function Base.run(spl::Sampler{PG})
  t_start = time()  # record the start time of PG
  chain = Chain()
  logevidence = Vector{Float64}(spl.alg.n_iterations)

  ## custom resampling function for pgibbs
  ## re-inserts reteined particle after each resampling step
  ref_particle = nothing
  for tt = 1:spl.alg.n_iterations
    spl.particles = ParticleContainer{TraceR}(spl.model)
    if ref_particle == nothing
      push!(spl.particles, spl.alg.n_particles)
    else
      push!(spl.particles, spl.alg.n_particles-1)
      push!(spl.particles, ref_particle)
    end

    while consume(spl.particles) != Val{:done}
      ess = effectiveSampleSize(spl.particles)
      if ess <= spl.alg.resampler_threshold * length(spl.particles)
        resample!(spl.particles, spl.alg.resampler, ref_particle)
      end
    end

    logevidence[tt] = spl.particles.logE
    ## pick a particle to be retained.
    Ws, _ = weights(spl.particles)
    indx = rand(Categorical(Ws))
    ref_particle = fork2(spl.particles[indx])

    s = getsample(spl.particles, indx)
    push!(chain, Sample(1/spl.alg.n_iterations, s.value))
  end
  chain.weight = exp(mean(logevidence))
  println("[PG]: Finshed within $(time() - t_start) seconds")
  chain
end
