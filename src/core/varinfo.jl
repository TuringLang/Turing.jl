export Var, VarInfo

########## Var ##########

doc"""
    Var(sym)

A wrapper of symbol type representing priors.

  - type is encoded in the form of [is_uni?, is_multi?, is_matrix?]

Usage:

```julia
p = Var(:somesym)
strp = string(p)
```
"""
immutable Var
  sym   ::    Symbol
  uid   ::    Symbol
  Var(sym::Symbol) = new(sym, sym)
  Var(sym::Symbol, uid::Symbol) = new(sym, uid)
end

doc"""
    Base.string(p::Var)

Helper function to convert a Var to its string representation.
"""
function Base.string(v::Var)
  return string(v.uid)
end

########## VarInfo ##########

doc"""
    VarInfo()

A values to store priors based on dictionary.

This type is basically a dictionary supporting adding new priors by creating a VarArray and indexing using pc[] syntax.

Usage:

```julia
pc = VarInfo()
p1 = Var(:a)
p2 = Var(:b)

addVar(pc, p1, 1)
addVar(pc, p1, 2)
addVar(pc, p1, 3)
addVar(pc, p2, 4)

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

keys(pc)  # create a key interator in the values, i.e. all the priors
```
"""
type VarInfo
  values      ::    Dict{Var, Any}
  dists       ::    Dict{Var, Distribution}
  logjoint    ::    Dual
  VarInfo() = new(Dict{Var, Any}(), Dict{Var, Distribution}(), Dual(0))
end

doc"""
    Base.getindex(pc::VarInfo, idx::Var)

Make the prior values support indexing with `[]`.
"""
function Base.getindex(pc::VarInfo, idx::Var)
  @assert haskey(pc.values, idx) "VarInfo has no $idx."
  return pc.values[idx]
end

doc"""
    Base.setindex!(pc::VarInfo, val, idx::Var)

Make the prior values support assignment with `[]`.
"""
function Base.setindex!(pc::VarInfo, val, idx::Var)
  @assert haskey(pc.values, idx) "VarInfo has no $idx."
  pc.values[idx] = val
end

doc"""
    Base.keys(vi::VarInfo)

Return a key interator in the values, i.e. all the priors.
"""
Base.keys(vi::VarInfo) = keys(vi.values)

doc"""
    syms(vi::VarInfo)

Return a set of all symbols
"""
syms(vi::VarInfo) = Set(map(v -> v.sym, keys(vi)))

export syms
