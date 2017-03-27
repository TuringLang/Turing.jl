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
type Chain <: Mamba.AbstractChains
  weight :: Float64 # log model evidence
  value2 :: Array{Sample}
  value::Array{Float64, 3}
  range::Range{Int}
  names::Vector{AbstractString}
  chains::Vector{Int}
end

Chain() = Chain(0, Vector{Sample}(), Array{Float64, 3}(0,0,0), 0:0,
                Vector{AbstractString}(), Vector{Int}())

Chain(w::Real, s::Array{Sample}) = begin
  chn = Chain()
  chn.weight = w
  chn.value2 = deepcopy(s)

  chn = flatten!(chn)
end

flatten!(chn::Chain) = begin
  ## Flatten samples into Mamba's chain type.
  local names = Array{Array{AbstractString}}(0)
  local vals  = Array{Array}(0)
  for s in chn.value2
    v, n = flatten(s)
    push!(vals, v)
    push!(names, n)
  end

  # Assuming that names[i] == names[j] for all (i,j)
  vals2 = [v[i] for v in vals, i=1:length(names[1])]
  vals2 = reshape(vals2, length(vals), length(names[1]), 1)
  c = Mamba.Chains(vals2, names = names[1])
  chn.value = c.value
  chn.range = c.range
  chn.names = c.names
  chn.chains = c.chains
  chn
end

flatten(s::Sample) = begin
  vals  = Array{Float64}(0)
  names = Array{AbstractString}(0)
  for (k, v) in s.value
    flatten(names, vals, string(k), v)
  end
  return vals, names
end
flatten(names, value :: Array{Float64}, k :: String, v) = begin
    if isa(v, Number)
      name = k
      push!(value, v)
      push!(names, name)
    elseif isa(v, Array)
      for i = eachindex(v)
        if isa(v[i], Number)
          name = k * string(ind2sub(size(v), i))
          name = replace(name, "(", "[");
          name = replace(name, ",)", "]");
          name = replace(name, ")", "]");
          isa(v[i], Void) && println(v, i, v[i])
          push!(value, Float64(v[i]))
          push!(names, name)
        elseif isa(v[i], Array)
          name = k * string(ind2sub(size(v), i))
          flatten(names, value, name, v[i])
        else
          error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
        end
      end
  else
    error("Unknown var type: typeof($v)=$(typeof(v))")
  end
end

function Base.getindex(c::Chain, v::Symbol)
  # This strange implementation is mostly to keep backward compatability.
  #  Needs some refactoring a better format for storing results is available.
  if v == :logevidence
    log(c.weight)
  elseif v==:samples
    c.value2
  elseif v==:logweights
    map((s)->s.weight, c.value2)
  else
    map((s)->Base.getindex(s, v), c.value2)
  end
end

function Base.vcat(c1::Chain, args::Chain...)

  names = c1.names
  all(c -> c.names == names, args) ||
    throw(ArgumentError("chain names differ"))

  chains = c1.chains
  all(c -> c.chains == chains, args) ||
    throw(ArgumentError("sets of chains differ"))

  value2 = cat(1, c1.value2, map(c -> c.value2, args)...)
  Chain(0, value2)
end

## NOTE: depreciated functions
# Base.push!(c::Chain, s::Sample) = push!(c.value2, s) #
# compute mean(f(x).w), where (x, w) is a weighted sample
# Base.mean(c::Chain, v::Symbol, f::Function) = mapreduce((s)->f(s[v]).*s.weight, +, c.value2)

# tests
# tr = Turing.sampler.particles[1]
# tr = Chain(Turing.sampler.particles)
