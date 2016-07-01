immutable IS <: InferenceAlgorithm
  n_particles :: Int
  IS(n) = new(n)
end

assume(spl :: Sampler{IS}, distr :: Distribution)  = rand(current_trace(), distr)

function Base.run(spl :: Sampler{IS})

  spl.particles = ParticleContainer{TraceC}(spl.model)
  push!(spl.particles, spl.alg.n_particles)

  while consume(spl.particles) != Val{:done}

  end

  res = Chain(spl.particles)
end
