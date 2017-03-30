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

function step(model, data, spl::Sampler{PG}, vi, ref_particle)
  spl.particles = ParticleContainer{TraceR}(spl.model)
  if ref_particle == nothing
    push!(spl.particles, spl.alg.n_particles, data, spl, vi)
  else
    push!(spl.particles, spl.alg.n_particles-1, data, spl, vi)
    push!(spl.particles, ref_particle)
  end

  while consume(spl.particles) != Val{:done}
    ess = effectiveSampleSize(spl.particles)
    if ess <= spl.alg.resampler_threshold * length(spl.particles)
      resample!(spl.particles, spl.alg.resampler, ref_particle)
    end
  end

  ## pick a particle to be retained.
  Ws, _ = weights(spl.particles)
  indx = rand(Categorical(Ws))
  ref_particle = fork2(spl.particles[indx])
  s = getsample(spl.particles, indx)
  ref_particle, s
end

function assume(spl::ParticleSampler{PG}, dist::Distribution, vn::VarName, vi::VarInfo)
  vi = current_trace().vi
  if spl == nothing || isempty(spl.alg.space) || vn.sym in spl.alg.space
    randrc(vi, vn, dist)
  else
    local r
    if ~haskey(vi, vn)
      dprintln(2, "sampling prior...")
      r = rand(dist)
      val = vectorize(dist, link(dist, r))      # X -> R and vectorize
      addvar!(vi, vn, val, dist)
    else
      dprintln(2, "fetching vals...")
      val = vi[vn]
      r = invlink(dist, reconstruct(dist, val)) # R -> X and reconstruct
    end
    produce(log(1.0))
    r
  end
end

function Base.run(model, data, spl::Sampler{PG})
  n = spl.alg.n_iterations
  t_start = time()  # record the start time of PG
  samples = Vector{Sample}()
  logevidence = Vector{Float64}(n)

  ## custom resampling function for pgibbs
  ## re-inserts reteined particle after each resampling step
  ref_particle = nothing
  for i = 1:n
    ref_particle, s = step(model, data, spl, VarInfo(), ref_particle)
    logevidence[i] = spl.particles.logE
    push!(samples, Sample(1/n, s.value))
  end

  println("[PG]: Finshed within $(time() - t_start) seconds")
  chain = Chain(exp(mean(logevidence)), samples)
end
