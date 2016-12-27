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
  function Var(sym::Symbol)
    new(sym, sym)
  end
  function Var(sym::Symbol, uid::Symbol)
    new(sym, uid)
  end
  function Var(sym::Symbol, arrExpr::Expr, idxSym::Symbol, idxVal::Any)
    if isa(arrExpr.args[2], Symbol)
      @assert arrExpr.args[2] == idxSym
      arrExpr.args[2] = idxVal
    end
    new(sym, Symbol(arrExpr))
  end
  function Var(sym::Symbol, mulDimExpr::Expr, dim1Sym::Symbol, dim1Val::Any, dim2Sym::Symbol, dim2Val::Any)
    if isa(mulDimExpr.args[1], Symbol)    # mat form x[i, j]
      if isa(mulDimExpr.args[2], Symbol)
        @assert mulDimExpr.args[2] == dim1Sym
        mulDimExpr.args[2] = dim1Val
      end
      if isa(mulDimExpr.args[3], Symbol)
        @assert mulDimExpr.args[3] == dim2Sym
        mulDimExpr.args[3] = dim2Val
      end
    elseif isa(mulDimExpr.args[1], Expr)  # multi array form x[i][j]
      if isa(mulDimExpr.args[1], Expr)
        @assert mulDimExpr.args[1].args[2] == dim1Sym
        mulDimExpr.args[1].args[2] = dim1Val
      end
      if isa(mulDimExpr.args[2], Symbol)
        @assert mulDimExpr.args[2] == dim2Sym
        mulDimExpr.args[2] = dim2Val
      end
    end
    new(sym, Symbol(replace(string(mulDimExpr), r"\(|\)", "")))
  end
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
  logjoint    ::    Dual
  function VarInfo()
    values = Dict{Var, Any}()
    new(values, Dual(0))
  end
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
