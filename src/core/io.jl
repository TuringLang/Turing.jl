# Some utility functions for extracting results.

type Sample
  weight :: Float64 # particle weight
  value :: Dict{Symbol,Any}
end

Base.getindex(s::Sample, v::Symbol) = Base.getindex(s.value, v)

type Chain
  weight :: Float64 # log model evidence
  value :: Array{Sample}
end
Chain() = Chain(0, Vector{Sample}())

function Base.getindex(c::Chain, v::Symbol)
  # This strange implementation is mostly to keep backward compatability.
  #  Needs some refactoring a better format for storing results is available.
  if v == :logevidence
    log(c.weight)
  else
    map((s)->Base.getindex(s, v), c.value)
  end
end

Base.push!(c::Chain, s::Sample) = push!(c.value, s)
# compute mean(f(x).w), where (x, w) is a weighted sample
Base.mean(c::Chain, v::Symbol, f::Function) = mapreduce((s)->f(s[v]).*s.weight, +, c.value)

Base.keys(p :: Particle) = keys(p.task.storage[:turing_predicts])
Base.values(p :: Particle) = values(p.task.storage[:turing_predicts])
Base.getindex(p :: Particle, args...) = getindex(p.task.storage[:turing_predicts], args...)

# ParticleContainer: particles ==> (weight, results)
function getsample(pc :: ParticleContainer, i :: Int64, w :: Float64 = 0.)
  p = pc.vals[i]

  predicts = Dict{Symbol, Any}()
  for k in keys(p)
    predicts[k] = p[k]
  end
  return Sample(w, predicts)
end

function Chain(pc :: ParticleContainer)
  w = pc.logE
  chain = Array{Sample}(length(pc))
  Ws, z = weights(pc)
  s = map((i)->chain[i] = getsample(pc, i, Ws[i]), 1:length(pc))

  Chain(exp(w), s)
end

# tests
# tr = Turing.sampler.particles[1]
# tr = Chain(Turing.sampler.particles)

