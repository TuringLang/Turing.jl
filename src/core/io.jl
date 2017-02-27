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


typealias TuringChains Chains
extract_sample!(names, value, k, v) = begin
    if isa(v, Number)
      name = string(k)
      push!(value, v)
      push!(names, name)
    elseif isa(v, Array)
      for i = eachindex(v)
        if isa(v[i], Number)
          name = string(k) * string(ind2sub(size(s), i))
          name = replace(name, "(", "["); name = replace(name, ")", "]")
          push!(value, v[i])
          push!(names, name)
        elseif isa(v[i], Array)
          extract_sample!(names, value, k, v[i])
        else
          error("Unknown var type: $(typeof(v[i]))")
        end
      end
  else
    error("Unknown var type: $(typeof(v))")
  end
end

TuringChains(chn::Chain) = begin
  # Get num of dimensions
  value_all = Array{Array}(0)
  names = Array{AbstractString}(0)
  for n = eachindex(chain.value)
    value = Array{Float64}(0)
    names = Array{AbstractString}(0)
    for (k, v) in chn.value[n].value
      extract_sample!(names, value, k, v)
    end
    push!(value_all, value)
  end
  value_all2 = [v[i] for v in value_all, i=1:length(names)]
  value_all2 = reshape(value_all2, length(value_all), length(names), 1)
  Chains(value_all2, names = names)
end


function Base.show(io::IO, ch1::Chain)
  # Print chain weight and weighted means of samples in chain
  if length(ch1.value) == 0
    print(io, "Empty Chain, weight $(ch1.weight)")
  elseif length(ch1.value) == 1
    chain_mean = Dict(i => mean(ch1, i, x -> x) for i in keys(ch1.value[1].value))
    print(io, "Chain, model evidence (log) $(ch1.weight) and means $(chain_mean)")
  else
    vars = keys(ch1.value[1].value)
    print(io, "Chain\nModel evidence (log) = $(ch1.weight)\n")
    for v in vars
      if isa(eltype(ch1[v]), Array)
        # TODO: implement support for array type.
        print(io, "Stats for array type not implemented yet.")
      else
        print(io, "Stats for :$v\n")
        stats = mcmcstats(ch1[v])
        for (label, value) in stats
          print(io, "  $label = $value\n")
        end
      end
    end
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

###############
# Declaration #
###############################################
# Code below are taken/adapted from           #
# Mamba.jl/blob/master/src/output/mcse.jl and #
# Mamba.jl/blob/master/src/output/stats.jl    #
###############################################
import StatsBase: sem

function mcse_bm{T<:Real}(x::Vector{T}; size::Integer=100)
  n = length(x)
  m = div(n, size)
  m >= 2 ||
    throw(ArgumentError(
      "iterations are < $(2 * size) and batch size is > $(div(n, 2))"
    ))
  mbar = [mean(x[i * size + (1:size)]) for i in 0:(m - 1)]
  sem(mbar)
end

# For univariate
function mcmcstats{T<:Real}(x::Vector{T})
  labels = ["Mean", "SD", "Naive SE", "MCSE", "ESS"]
  vals = [mean(x), std(x), sem(x), mcse_bm(x)]
  stats = [vals;  min((vals[2] / vals[4])^2, length(x))]
  Dict(labels[i] => stats[i] for i in 1:5)
end

# For multivariate
function mcmcstats{T<:Real}(x::Array{T, 2})
  labels = ["Mean", "SD", "Naive SE", "MCSE", "ESS"]
  f = x -> [mean(x), std(x), sem(x), mcse_bm(vec(x))]
  vals = permutedims(
    mapslices(x -> f(x), x, [1 2]),
    [2, 1]
  )
  stats = [vals  min((vals[:, 2] ./ vals[:, 4]).^2, size(x, 2))]
  Dict(labels[i] => stats[i] for i in 1:5)
end
