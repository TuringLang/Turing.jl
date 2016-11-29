export VarInfo, VarInfoArray, GradientInfo, addVarInfo

doc"""
    VarInfoArray(array, count, currSetIdx, currGetIdx)

Type for a rotated array (used for prior replay purpose).

This type of array support set and get without specifying indices. Instead, an inner index pointer is used to iterate the array. The pointers for set and get are separate.

Usage:

```julia
pa = VarInfoArray() # []
add(pa, 1)        # [1]
add(pa, 2)        # [1, 2]
get(pa)           # 1
get(pa)           # 2
set(pa, 3)        # [3, 2]
get(pa)           # 3
get(pa)           # 2
```
"""
type VarInfoArray
  array     ::    Vector{Any}
  count         ::    Int64
  currSetIdx    ::    Int64
  currGetIdx    ::    Int64
  function VarInfoArray()
    array = Vector{Any}()
    count = 0
    currSetIdx = 1
    currGetIdx = 1
    new(array, count, currSetIdx, currGetIdx)
  end
end

doc"""
    add(pa::VarInfoArray, val)

Append a new element to the end of the inner array container of VarInfoArray.
The inner counter of total number of element is also updated.
"""
function add(pa::VarInfoArray, val)
  push!(pa.array, val)
  pa.count += 1
end

doc"""
    set(pa::VarInfoArray, val)

Set the element with the corresponding inner pointer of set to the passed value.

The inner pointer for set is then updated:
  - if not reaches the end: incremented by 1;
  - if reaches the end: reset to 1.
"""
function set(pa::VarInfoArray, val)
  pa.array[pa.currSetIdx] = val
  pa.currSetIdx = (pa.currSetIdx + 1) > pa.count? 1 : pa.currSetIdx + 1 # rotate if reaches the end
end

doc"""
    get(pa::VarInfoArray)

Fetch the element with the corresponding inner pointer of get.

The inner pointer for get is then updated:
  - if not reaches the end: incremented by 1;
  - if reaches the end: reset to 1.
"""
function get(pa::VarInfoArray)
  @assert pa.count > 0 "Attempt get from an empty VarInfoArray."
  oldGetIdx = pa.currGetIdx   # record the old get index
  pa.currGetIdx = (pa.currGetIdx + 1) > pa.count? 1 : pa.currGetIdx + 1 # rotate if reaches the end
  return pa.array[oldGetIdx]
end



doc"""
    VarInfo(sym)

A wrapper of symbol type representing priors.

  - type is encoded in the form of [is_uni?, is_multi?, is_matrix?]

Usage:

```julia
p = VarInfo(:somesym)
strp = string(p)
```
"""
immutable VarInfo
  sym       ::    Symbol
  name      ::    Symbol
  typ       ::    Int64
  function VarInfo(sym)
    new(sym, :unknownname, 0)
  end
  function VarInfo(sym, name, typ)
    new(sym, name, typ)
  end
end

doc"""
    Base.string(p::VarInfo)

Helper function to convert a VarInfo to its string representation.
"""
function Base.string(p::VarInfo)
  return string(p.sym)
end



doc"""
    GradientInfo()

A container to store priors based on dictionary.

This type is basically a dictionary supporting adding new priors by creating a VarInfoArray and indexing using pc[] syntax.

Usage:

```julia
pc = GradientInfo()
p1 = VarInfo(:a)
p2 = VarInfo(:b)

addVarInfo(pc, p1, 1)
addVarInfo(pc, p1, 2)
addVarInfo(pc, p1, 3)
addVarInfo(pc, p2, 4)

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
"""
type GradientInfo
  container   ::    Dict{VarInfo, VarInfoArray}
  logjoint    ::    Dual
  function GradientInfo()
    container = Dict{VarInfo, VarInfoArray}()
    new(container, Dual(0))
  end
end

doc"""
    addVarInfo(pc::GradientInfo, idx::VarInfo, val)

Add a *new* value of a given prior to the container.
*new* here means force appending to the end of the corresponding array of the prior.
"""
function addVarInfo(pc::GradientInfo, idx::VarInfo, val)
  if haskey(pc.container, idx)
    add(pc.container[idx], val)
  else
    pc.container[idx] = VarInfoArray()  # create a PA if new
    add(pc.container[idx], val)
  end
end

doc"""
    Base.getindex(pc::GradientInfo, idx::VarInfo)

Make the prior container support indexing with `[]`.
"""
function Base.getindex(pc::GradientInfo, idx::VarInfo)
  @assert haskey(pc.container, idx) "GradientInfo has no $idx."
  return get(pc.container[idx])
end

doc"""
    Base.setindex!(pc::GradientInfo, val, idx::VarInfo)

Make the prior container support assignment with `[]`.
"""
function Base.setindex!(pc::GradientInfo, val, idx::VarInfo)
  @assert haskey(pc.container, idx) "GradientInfo has no $idx."
  set(pc.container[idx], val)
end

doc"""
    Base.keys(pc::GradientInfo)

Return a key interator in the container, i.e. all the priors.
"""
function Base.keys(pc::GradientInfo)
  return keys(pc.container)
end
