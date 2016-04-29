"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""

typealias Particle Trace

type ParticleContainer{T<:Particle} <: DenseArray{T,1}
  model :: Function
  num_particles :: Int64
  particles :: Dict{T, Float64} # (Trace, Log Weight)
  logevidence :: Float64
  incremental_logliklihood :: Float64   # log weight of theta particle (useful for SMC2)
  ParticleContainer(m::Function,n::Int64) = new(m,n,Dict{Particle, Float64}(),0,0)
end

call{T}(::Type{ParticleContainer{T}}, m) = ParticleContainer{T}(m, 0)

Base.collect(pc :: ParticleContainer) = pc.particles
Base.length(pc :: ParticleContainer)  = length(pc.particles)
Base.similar(pc :: ParticleContainer) = ParticleContainer(pc.model, 0)
# pc[i] returns the i'th particle
Base.getindex(pc :: ParticleContainer, i :: Real) = collect(keys(pc.particles))[i]


# registers a new x-particle in the container
function Base.push!(pc :: ParticleContainer, p :: Particle)
  pc.num_particles += 1
  pc.particles[p] = 0
  pc
end
Base.push!(pc :: ParticleContainer) = Base.push!(pc, keytype(pc.particles)(pc.model))
Base.push!(pc :: ParticleContainer, n :: Int64) = map((i)->Base.push!(pc), 1:n)

# clears the container but keep params, logweight etc.
function Base.pop!(pc :: ParticleContainer)
  pc.num_particles = 0
  pc.particles = Dict{Particle, Float64}()
end

# clones a theta-particle
function Base.copy(pc :: ParticleContainer)
  particles = collect(pc)
  newpc     = similar(pc)
  for (p, v) in particles
    newp = forkc(p)
    push!(newpc, newp)
  end
  newpc.incremental_logliklihood = pc.incremental_logliklihood
  newpc.logevidence      = pc.logevidence
end

# run particle filter for one step, return incremental likelihood
function Base.consume(pc :: ParticleContainer)
  @assert pc.num_particles == length(pc)
  # normalisation factor: 1/N
  _, z1      = weights(pc)

  particles = collect(pc)
  traces    = keys(particles)
  num_done = 0
  for p = traces
    running = +1
    score = consume(p)
    if isa(score, Real)
      increase_logweight(pc, p, Float64(score))
    elseif score == Val{:done}
      num_done += 1
    else
      "[consume]: error in running particle filter."
    end
  end

  if num_done == length(pc)
    res = Val{:done}
  elseif num_done != 0
    error("[consume]: mis-aligned execution traces.")
  else
    # update incremental likelihoods
    _, z2      = weights(pc)
    increase_logevidence(pc, z2 - z1)
    res = increase_loglikelihood(pc, z2 - z1)
  end

  res
end

function weights(pc :: ParticleContainer)
  @assert pc.num_particles == length(pc)
  logWs = collect(values(pc.particles))
  Ws = exp(logWs-maximum(logWs))
  z = log(sum(Ws)) + maximum(logWs)
  Ws = Ws ./ sum(Ws)
  return Ws, z
end

function effectiveSampleSize(pc :: ParticleContainer)
  Ws, _ = weights(pc)
  ess = sum(Ws) ^ 2 / sum(Ws .^ 2)
end

increase_logweight(pc :: ParticleContainer, t :: Particle, logw :: Float64) =
  (pc.particles[t]  += logw)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
  (pc.logevidence += logw)

increase_loglikelihood(pc :: ParticleContainer, logw :: Float64) =
  (pc.incremental_logliklihood += logw)


function resample!( pc :: ParticleContainer,
                   randcat :: Function = Turing.resampleSystematic,
                   ref :: Union{Particle, Void} = nothing)
  n1, particles = pc.num_particles, collect(pc)
  @assert n1 == length(particles)

  # resample
  Ws, _ = weights(pc)
  n2    = isa(ref, Void) ? n1 : n1-1
  indx  = randcat(Ws, n2)

  # fork particles
  pop!(pc)
  traces = collect(keys(particles))
  for i = 1:length(indx)
    tr = traces[indx[i]]
    newtrace = TraceM.forkc(tr)
    push!(pc, newtrace)
  end

  if isa(ref, Particle)
    # Insert the retained particle. This is based on the replaying trick for efficiency
    #  reasons. If we implement PG using task copying, we need to store Nx * T particles!
    push!(pc, ref)
  end

  pc
end





