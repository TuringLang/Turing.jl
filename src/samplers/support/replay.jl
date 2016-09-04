# This type Prior is for passing Symbol/Array priors to HMc sampler.
# The corresponding parameter in @assume is only used by the HMC sampler.



export Prior, PriorArray, PriorContainer



type PriorArray
  array     ::    Vector{Any}
  count         ::    Int64
  currSetIdx    ::    Int64
  currGetIdx    ::    Int64
  add           ::    Function
  set           ::    Function
  get           ::    Function
  function PriorArray()
    array = Vector{Any}()
    count = 0
    currSetIdx = 1
    currGetIdx = 1
    function add(val)
      push!(array, val)
      count += 1
    end
    function set(val)
      array[currSetIdx] = val
      currSetIdx = (currSetIdx + 1) > count? 1 : currSetIdx + 1     # rotate
    end
    function get()
      @assert count > 0 "Attempt get from an empty PriorArray."
      oldGetIdx = currGetIdx
      currGetIdx = (currGetIdx + 1) > count? 1 : currGetIdx + 1   # rotate
      return array[oldGetIdx]
    end
    new(array, count, currSetIdx, currGetIdx, add, set, get)
  end
end

immutable Prior
  sym       ::    Symbol
  function Prior(sym)
    new(sym)
  end
end

function Base.string(p::Prior)
  return string(p.sym)
end

type PriorContainer
  container   ::    Dict{Prior, PriorArray}
  addPrior    ::    Function
  function PriorContainer()
    container = Dict{Prior, PriorArray}()
    function addPrior(idx::Prior, val)
      if haskey(container, idx)
        container[idx].add(val)
      else
        container[idx] = PriorArray()
        container[idx].add(val)
      end
    end
    new(container, addPrior)
  end
end

function Base.getindex(pc::PriorContainer, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  return pc.container[idx].get()
end

function Base.setindex!(pc::PriorContainer, val, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  pc.container[idx].set(val)
end

function Base.keys(pc::PriorContainer)
  return keys(pc.container)
end
