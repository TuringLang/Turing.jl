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
  n_particles           ::    Int
  n_iterations          ::    Int
  resampler             ::    Function
  resampler_threshold   ::    Float64
  space                 ::    Set
  group_id              ::    Int
  PG(n1::Int, n2::Int) = new(n1, n2, resampleSystematic, 0.5, Set(), 0)
  function PG(n1::Int, n2::Int, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n1, n2, resampleSystematic, 0.5, space, 0)
  end
  PG(alg::PG, new_group_id::Int) = new(alg.n_particles, alg.n_iterations, alg.resampler, alg.resampler_threshold, alg.space, new_group_id)
end

function step(model, spl::Sampler{PG}, vi, ref_particle)
  spl.particles = ParticleContainer{TraceR}(model)
  if ref_particle == nothing
    push!(spl.particles, spl.alg.n_particles, spl, vi)
  else
    push!(spl.particles, spl.alg.n_particles-1, spl, vi)
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

sample(model, alg::PG) = begin
  global sampler = ParticleSampler{PG}(alg);
  spl = sampler
  n = spl.alg.n_iterations
  samples = Vector{Sample}()
  logevidence = Vector{Float64}(n)

  ## custom resampling function for pgibbs
  ## re-inserts reteined particle after each resampling step
  ref_particle = nothing
  @showprogress 1 "[PG] Sampling..." for i = 1:n
    ref_particle, s = step(model, spl, VarInfo(), ref_particle)
    logevidence[i] = spl.particles.logE
    push!(samples, Sample(1/n, s.value))
  end

  chain = Chain(exp(mean(logevidence)), samples)
end

assume(spl::ParticleSampler{PG}, d::Distribution, vn::VarName, vi::VarInfo) = begin
  rand(current_trace().vi, vn, d, spl)
end

rand(vi::VarInfo, vn::VarName, d::Distribution, spl::ParticleSampler{PG}) = begin
  isempty(spl.alg.space) || vn.sym in spl.alg.space ?
    randr(vi, vn, d, spl, true) :
    randr(vi, vn, d)
end
