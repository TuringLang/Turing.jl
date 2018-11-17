const Particle = Trace

"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T<:Particle}
  model :: Function
  num_particles :: Int
  vals  :: Array{T}
  logWs :: Array{Float64}  # Log weights (Trace) or incremental likelihoods (ParticleContainer)
  logE  :: Float64           # Log model evidence
  # conditional :: Union{Nothing,Conditional} # storing parameters, helpful for implementing rejuvenation steps
  conditional :: Nothing # storing parameters, helpful for implementing rejuvenation steps
  n_consume :: Int # helpful for rejuvenation steps, e.g. in SMC2
end
ParticleContainer{T}(m) where T = ParticleContainer{T}(m, 0)
function ParticleContainer{T}(m::Function,n::Int) where T
  ParticleContainer{T}(m, n, Vector{T}(), Vector{Float64}(), 0.0, nothing, 0)
end

Base.collect(pc :: ParticleContainer) = pc.vals # prev: Dict, now: Array
Base.length(pc :: ParticleContainer)  = length(pc.vals)
Base.similar(pc :: ParticleContainer{T}) where T = ParticleContainer{T}(pc.model, 0)
# pc[i] returns the i'th particle
Base.getindex(pc :: ParticleContainer, i :: Real) = pc.vals[i]


# registers a new x-particle in the container
function Base.push!(pc :: ParticleContainer, p :: Particle)
  pc.num_particles += 1
  push!(pc.vals, p)
  push!(pc.logWs, 0)
  pc
end
Base.push!(pc :: ParticleContainer) = Base.push!(pc, eltype(pc.vals)(pc.model))

function Base.push!(pc :: ParticleContainer, n :: Int, spl :: Sampler, varInfo :: VarInfo)
  vals  = Vector{eltype(pc.vals)}(undef,n)
  logWs = zeros(eltype(pc.logWs), n)
  for i=1:n
    vals[i]  = eltype(pc.vals)(pc.model, spl, varInfo)
  end
  append!(pc.vals, vals)
  append!(pc.logWs, logWs)
  pc.num_particles += n
  pc
end

# clears the container but keep params, logweight etc.
function Base.empty!(pc :: ParticleContainer)
  pc.num_particles = 0
  pc.vals  = Vector{Particle}()
  pc.logWs = Vector{Float64}()
  pc
end

# clones a theta-particle
function Base.copy(pc :: ParticleContainer)
  particles = collect(pc)
  newpc     = similar(pc)
  for p in particles
    newp = fork(p)
    push!(newpc, newp)
  end
  newpc.logE        = pc.logE
  newpc.logWs       = deepcopy(pc.logWs)
  newpc.conditional = deepcopy(pc.conditional)
  newpc.n_consume   = pc.n_consume
  newpc
end

# run particle filter for one step, return incremental likelihood
function Turing.consume(pc :: ParticleContainer)
  @assert pc.num_particles == length(pc)
  # normalisation factor: 1/N
  _, z1      = weights(pc)
  n = length(pc.vals)

  particles = collect(pc)
  num_done = 0
  for i=1:n
    p = pc.vals[i]
    score = Turing.consume(p)
    if score isa Real
      score += getlogp(p.vi)
      resetlogp!(p.vi)
      increase_logweight(pc, i, Float64(score))
    elseif score == Val{:done}
      num_done += 1
    else
      error("[consume]: error in running particle filter.")
    end
  end

  if num_done == length(pc)
    res = Val{:done}
  elseif num_done != 0
    error("[consume]: mis-aligned execution traces, num_particles= $(n), num_done=$(num_done).")
  else
    # update incremental likelihoods
    _, z2      = weights(pc)
    res = increase_logevidence(pc, z2 - z1)
    pc.n_consume += 1
    # res = increase_loglikelihood(pc, z2 - z1)
  end

  res
end

function weights(pc :: ParticleContainer)
  @assert pc.num_particles == length(pc)
  logWs = pc.logWs
  Ws = exp.(logWs .- maximum(logWs))
  logZ = log(sum(Ws)) + maximum(logWs)
  Ws = Ws ./ sum(Ws)
  return Ws, logZ
end

function effectiveSampleSize(pc :: ParticleContainer)
  Ws, _ = weights(pc)
  ess = 1.0 / sum(Ws .^ 2) # sum(Ws) ^ 2 = 1.0, because weights are normalised
end

increase_logweight(pc :: ParticleContainer, t :: Int, logw :: Float64) =
  (pc.logWs[t]  += logw)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
  (pc.logE += logw)


function resample!( pc :: ParticleContainer,
                   randcat :: Function = Turing.resample_systematic,
                   ref :: Union{Particle, Nothing} = nothing)
  n1, particles = pc.num_particles, collect(pc)
  @assert n1 == length(particles)

  # resample
  Ws, _ = weights(pc)

  # check that weights are not NaN
  @assert !any(isnan.(Ws))

  n2    = isa(ref, Nothing) ? n1 : n1-1
  indx  = randcat(Ws, n2)

  # fork particles
  empty!(pc)
  num_children = zeros(Int,n1)
  map(i->num_children[i]+=1, indx)
  for i = 1:n1
    is_ref = particles[i] == ref
    p = is_ref ? fork(particles[i], is_ref) : particles[i]
    num_children[i] > 0 && push!(pc, p)
    for k=1:num_children[i]-1
      newp = fork(p, is_ref)
      push!(pc, newp)
    end
  end

  if isa(ref, Particle)
    # Insert the retained particle. This is based on the replaying trick for efficiency
    #  reasons. If we implement PG using task copying, we need to store Nx * T particles!
    push!(pc, ref)
  end

  pc
end


########### Auxilary Functions ###################

# ParticleContainer: particles ==> (weight, results)
function getsample(pc :: ParticleContainer, i :: Int, w :: Float64 = 0.)
  p = pc.vals[i]
  predicts = Sample(p.vi).value
  predicts[:le] = pc.logE
  return Sample(w, predicts)
end

getsample(pc :: ParticleContainer) = begin
  w = pc.logE
  Ws, z = weights(pc)
  s = map((i)->getsample(pc, i, Ws[i]), 1:length(pc))
  return exp.(w), s
end
