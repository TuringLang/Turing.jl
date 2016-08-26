# This type Prior is for passing Symbol/Array priors to HMc sampler.
# The corresponding parameter in @assume is only used by the HMC sampler.



export Prior, PriorArray, PriorContainer, addPrior



type PriorArray
  array     ::    Vector{Any}
  count         ::    Int64
  currSetIdx    ::    Int64
  currGetIdx    ::    Int64
  function PriorArray()
    array = Vector{Any}()
    count = 0
    currSetIdx = 1
    currGetIdx = 1
    new(array, count, currSetIdx, currGetIdx)
  end
end

function add(pa::PriorArray, val)
  push!(pa.array, val)
  pa.count += 1
end
function set(pa::PriorArray, val)
  pa.array[pa.currSetIdx] = val
  pa.currSetIdx = (pa.currSetIdx + 1) > pa.count? 1 : pa.currSetIdx + 1
end
function get(pa::PriorArray)
  @assert pa.count > 0 "Attempt get from an empty PriorArray."
  oldGetIdx = pa.currGetIdx
  pa.currGetIdx = (pa.currGetIdx + 1) > pa.count? 1 : pa.currGetIdx + 1
  return pa.array[oldGetIdx]
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
  function PriorContainer()
    container = Dict{Prior, PriorArray}()
    new(container)
  end
end

function addPrior(pc::PriorContainer, idx::Prior, val)
  if haskey(pc.container, idx)
    add(pc.container[idx], val)
  else
    pc.container[idx] = PriorArray()
    add(pc.container[idx], val)
  end
end

function Base.getindex(pc::PriorContainer, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  return get(pc.container[idx])
end

function Base.setindex!(pc::PriorContainer, val, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  set(pc.container[idx], val)
end

function Base.keys(pc::PriorContainer)
  return keys(pc.container)
end
