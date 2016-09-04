export Prior, PriorArray, PriorContainer, addPrior



# Type for a rotated array.
# This type of array support set and get without specifying indices. Instead, a inner index counter will iterate the array.
# The set and get counter are separate.
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
  pa.currSetIdx = (pa.currSetIdx + 1) > pa.count? 1 : pa.currSetIdx + 1 # rotate if reaches the end
end
function get(pa::PriorArray)
  @assert pa.count > 0 "Attempt get from an empty PriorArray."
  oldGetIdx = pa.currGetIdx   # record the old get index
  pa.currGetIdx = (pa.currGetIdx + 1) > pa.count? 1 : pa.currGetIdx + 1 # rotate if reaches the end
  return pa.array[oldGetIdx]
end



# A wrapper of symbol type representing priors
immutable Prior
  sym       ::    Symbol
  function Prior(sym)
    new(sym)
  end
end

function Base.string(p::Prior)
  return string(p.sym)
end



# A container to store priors based on dictionary
# This type is basically a dictionary supporting adding new priors by creating a PriorArray and indexing using pc[] syntax
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
    pc.container[idx] = PriorArray()  # create a PA if new
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
