export Prior, PriorArray, PriorContainer, addPrior

doc"""
  Type for a rotated array (used for prior replay purpose).

  This type of array support set and get without specifying indices. Instead, an inner index pointer is used to iterate the array. The pointers for set and get are separate.

  Usage:

  ```julia
  pa = PriorArray() # []
  add(pa, 1)        # [1]
  add(pa, 2)        # [1, 2]
  get(pa)           # 1
  get(pa)           # 2
  set(pa, 3)        # [3, 2]
  get(pa)           # 3
  get(pa)           # 2
  ```

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
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

doc"""
  Append a new element to the end of the inner array container of PriorArray.
  The inner counter of total number of element is also updated.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function add(pa::PriorArray, val)
  push!(pa.array, val)
  pa.count += 1
end

doc"""
  Set the element with the corresponding inner pointer of set to the passed value.

  The inner pointer for set is then updated:
    - if not reaches the end: incremented by 1;
    - if reaches the end: reset to 1.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function set(pa::PriorArray, val)
  pa.array[pa.currSetIdx] = val
  pa.currSetIdx = (pa.currSetIdx + 1) > pa.count? 1 : pa.currSetIdx + 1 # rotate if reaches the end
end

doc"""
  Fetch the element with the corresponding inner pointer of get.

  The inner pointer for get is then updated:
    - if not reaches the end: incremented by 1;
    - if reaches the end: reset to 1.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function get(pa::PriorArray)
  @assert pa.count > 0 "Attempt get from an empty PriorArray."
  oldGetIdx = pa.currGetIdx   # record the old get index
  pa.currGetIdx = (pa.currGetIdx + 1) > pa.count? 1 : pa.currGetIdx + 1 # rotate if reaches the end
  return pa.array[oldGetIdx]
end



doc"""
  A wrapper of symbol type representing priors.

  Usage:

  ```julia
  p = Prior(:somesym)
  strp = string(p)
  ```

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
immutable Prior
  sym       ::    Symbol
  function Prior(sym)
    new(sym)
  end
end

doc"""
  Helper function to convert a Prior to its string representation.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function Base.string(p::Prior)
  return string(p.sym)
end

doc"""
  A container to store priors based on dictionary.

  This type is basically a dictionary supporting adding new priors by creating a PriorArray and indexing using pc[] syntax

  Usage:

  ```julia
  pc = PriorContainer()
  p1 = Prior(:a)
  p2 = Prior(:b)

  addPrior(pc, p1, 1)
  addPrior(pc, p1, 2)
  addPrior(pc, p1, 3)
  addPrior(pc, p2, 4)

  pc[p1]    # 1
  pc[p1]    # 2
  pc[p1]    # 3
  pc[p1]    # 1
  pc[p1]    # 2
  pc[p1]    # 3

  pc[p2]    # 4

  pc[p1] = 5
  pc[p1] = 6
  pc[p1] = 7

  pc[p1]    # 5
  pc[p1]    # 6
  pc[p1]    # 7

  keys(pc)  # create a key interator in the container, i.e. all the priors
  ```

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
type PriorContainer
  container   ::    Dict{Prior, PriorArray}
  function PriorContainer()
    container = Dict{Prior, PriorArray}()
    new(container)
  end
end

doc"""
  Add a *new* value of a given prior to the container.
  *new* here means force appending to the end of the corresponding array of the prior.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function addPrior(pc::PriorContainer, idx::Prior, val)
  if haskey(pc.container, idx)
    add(pc.container[idx], val)
  else
    pc.container[idx] = PriorArray()  # create a PA if new
    add(pc.container[idx], val)
  end
end

doc"""
  Make the prior container support indexing with `[]`.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function Base.getindex(pc::PriorContainer, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  return get(pc.container[idx])
end

doc"""
  Make the prior container support assignment with `[]`.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function Base.setindex!(pc::PriorContainer, val, idx::Prior)
  @assert haskey(pc.container, idx) "PriorContainer has no $idx."
  set(pc.container[idx], val)
end

doc"""
  Return a key interator in the container, i.e. all the priors.

  --- Info ---

  Code location: src/samplers/support/replay.jl
"""
function Base.keys(pc::PriorContainer)
  return keys(pc.container)
end
