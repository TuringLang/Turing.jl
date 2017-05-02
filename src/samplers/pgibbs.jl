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
  gid                   ::    Int
  PG(n1::Int, n2::Int) = new(n1, n2, resampleSystematic, 0.5, Set(), 0)
  function PG(n1::Int, n2::Int, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n1, n2, resampleSystematic, 0.5, space, 0)
  end
  PG(alg::PG, new_gid::Int) = new(alg.n_particles, alg.n_iterations, alg.resampler, alg.resampler_threshold, alg.space, new_gid)
end

Sampler(alg::PG) = begin
  info = Dict{Symbol, Any}()
  info[:logevidence] = []
  Sampler(alg, info)
end

function step(model, spl::Sampler{PG}, vi, ref_particle)
  particles = ParticleContainer{TraceR}(model)
  if ref_particle == nothing
    push!(particles, spl.alg.n_particles, spl, vi)
  else
    push!(particles, spl.alg.n_particles-1, spl, vi)
    push!(particles, ref_particle)
  end

  while consume(particles) != Val{:done}
    ess = effectiveSampleSize(particles)
    if ess <= spl.alg.resampler_threshold * length(particles)
      resample!(particles, spl.alg.resampler, ref_particle)
    end
  end

  ## pick a particle to be retained.
  Ws, _ = weights(particles)
  indx = rand(Categorical(Ws))
  ref_particle = fork2(particles[indx])
  s = getsample(particles, indx)
  push!(spl.info[:logevidence], particles.logE)
  ref_particle, s
end

sample(model::Function, alg::PG) = begin
  spl = Sampler(alg);
  n = spl.alg.n_iterations
  samples = Vector{Sample}()

  ## custom resampling function for pgibbs
  ## re-inserts reteined particle after each resampling step
  ref_particle = nothing
  @showprogress 1 "[PG] Sampling..." for i = 1:n
    ref_particle, s = step(model, spl, VarInfo(), ref_particle)
    push!(samples, Sample(1/n, s.value))
  end

  chain = Chain(exp(mean(spl.info[:logevidence])), samples)
end

assume{T<:Union{PG,SMC}}(spl::Sampler{T}, dist::Distribution, vn::VarName, _::VarInfo) = begin
  vi = current_trace().vi
  if isempty(spl.alg.space) || vn.sym in spl.alg.space
    vi.index += 1
    if ~haskey(vi, vn)
      r = rand(dist)
      push!(vi, Var(vn, vectorize(dist, r), dist, spl.alg.gid))
      spl.info[:ranges_updated] = false
      r
    elseif isnan(vi, vn)
      r = rand(dist)
      setval!(vi, vectorize(dist, r), vn)
      r
    else
      checkindex(vn, vi, spl)
      updategid!(vi, vn, spl)
      vi[vn]
    end
  else
    vi[vn]
  end
end

observe{T<:Union{PG,SMC}}(spl::Sampler{T}, dist::Distribution, value, vi) = begin
  lp = logpdf(dist, value)
  vi.logp += lp
  produce(lp)
end
