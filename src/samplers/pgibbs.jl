# Particle Gibbs sampler

immutable PG <: InferenceAlgorithm
  n_particles :: Int
  n_iterations :: Int
  resampler :: Function
  resampler_threshold :: Float64
  PG(n1::Int,n2::Int) = new(n1,n2,resampleSystematic,0.5)
end

assume(spl :: Sampler{PG}, distr :: Distribution)   = randr(current_trace(), distr)

function Base.run(spl::Sampler{PG})
  chain = Chain()
  logevidence = Vector{Float64}(spl.alg.n_iterations)

  ## custom resampling function for pgibbs
  ## re-inserts reteined particle after each resampling step
  ref_particle = nothing
  for tt = 1:spl.alg.n_iterations
    dprintln(-1, "[PG]: Iter $(tt) out of $(spl.alg.n_iterations)")
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
  chain
end
