#########################
# Sampler I/O Interface #
#########################

##########
# Sample #
##########

doc"""
    Sample(weight::Float64, value::Dict{Symbol,Any})

A wrapper of output samples.

Example:

```julia
# Define a model
@model xxx begin
  ...
  @predict mu sigma
end

# Run the inference engine
chain = sample(xxx, SMC(1000))

sample = chain[:mu][1]  # get the first sample
sample.weight           # show the weight of this sample
sample.value            # show the value of this sample (a dictionary)
```
"""
type Sample
  weight :: Float64     # particle weight
  value :: Dict{Symbol,Any}
end

Base.getindex(s::Sample, v::Symbol) = Base.getindex(s.value, v)

#########
# Chain #
#########

doc"""
    Chain(weight::Float64, value::Array{Sample})

A wrapper of output trajactory of samplers.

Example:

```julia
# Define a model
@model xxx begin
  ...
  @predict mu sigma
end

# Run the inference engine
chain = sample(xxx, SMC(1000))

chain[:logevidence]   # show the log model evidence
chain[:mu]            # show the weighted trajactory for :mu
chain[:sigma]         # show the weighted trajactory for :sigma
mean(chain[:mu])      # find the mean of :mu
mean(chain[:sigma])   # find the mean of :sigma
```
"""
type Chain
  weight :: Float64 # log model evidence
  value :: Array{Sample}
end

Chain() = Chain(0, Vector{Sample}())

function Base.show(io::IO, ch1::Chain)
  # Print chain weight and weighted means of samples in chain
  if length(ch1.value) == 0
    print(io, "Empty Chain, weight $(ch1.weight)")
  else
    chain_mean = [i => mean(ch1, i, x -> x) for i in keys(ch1.value[1].value)]
    print(io, "Chain, model evidence (log)  $(ch1.weight) and means $(chain_mean)")
  end
end

function Base.getindex(c::Chain, v::Symbol)
  # This strange implementation is mostly to keep backward compatability.
  #  Needs some refactoring a better format for storing results is available.
  if v == :logevidence
    log(c.weight)
  elseif v==:samples
    c.value
  else
    map((s)->Base.getindex(s, v), c.value)
  end
end

Base.push!(c::Chain, s::Sample) = push!(c.value, s)
# compute mean(f(x).w), where (x, w) is a weighted sample
Base.mean(c::Chain, v::Symbol, f::Function) = mapreduce((s)->f(s[v]).*s.weight, +, c.value)

# NOTE: Particle is a type alias of Trace
Base.keys(p :: Particle) = keys(p.task.storage[:turing_predicts])
Base.values(p :: Particle) = values(p.task.storage[:turing_predicts])
Base.getindex(p :: Particle, args...) = getindex(p.task.storage[:turing_predicts], args...)

# ParticleContainer: particles ==> (weight, results)
function getsample(pc :: ParticleContainer, i :: Int, w :: Float64 = 0.)
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
