immutable SMC <: InferenceAlgorithm
  n_particles :: Int64
  resampler :: Function
  resampler_threshold :: Float64
  SMC(n) = new(n, resampleSystematic, 0.5)
end

assume(spl :: Sampler{SMC}, distr :: Distribution)  = rand(current_trace(), distr)

## wrapper for smc: run the sampler, collect results.
function Base.run(spl::Sampler{SMC})

  spl.particles = ParticleContainer{TraceC}(spl.model)
  push!(spl.particles, spl.alg.n_particles)

  while consume(spl.particles) != Val{:done}
    ess = effectiveSampleSize(spl.particles)
    if ess <= spl.alg.resampler_threshold * length(spl.particles)
      resample!(spl.particles)
    end
  end

  res = Chain(spl.particles)

end
