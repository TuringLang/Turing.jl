export VarInfo, GradientInfo

########## VarInfo ##########

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
  id    ::    Symbol
  function VarInfo(sym::Symbol)
    new(sym)
  end
  function VarInfo(arrExpr::Expr, idxSym::Symbol, idxVal::Any)
    if isa(arrExpr.args[2], Symbol)
      @assert arrExpr.args[2] == idxSym
      arrExpr.args[2] = idxVal
    end
    new(Symbol(arrExpr))
  end
  function VarInfo(mulDimExpr::Expr, dim1Sym::Symbol, dim1Val::Any, dim2Sym::Symbol, dim2Val::Any)
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
    new(Symbol(replace(string(mulDimExpr), r"\(|\)", "")))
  end
end

doc"""
    Base.string(p::VarInfo)

Helper function to convert a VarInfo to its string representation.
"""
function Base.string(p::VarInfo)
  return string(p.id)
end

########## GradientInfo ##########

doc"""
    GradientInfo()

A values to store priors based on dictionary.

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

keys(pc)  # create a key interator in the values, i.e. all the priors
```
"""
type GradientInfo
  values      ::    Dict{VarInfo, Any}
  logjoint    ::    Dual
  function GradientInfo()
    values = Dict{VarInfo, Any}()
    new(values, Dual(0))
  end
end

doc"""
    Base.getindex(pc::GradientInfo, idx::VarInfo)

Make the prior values support indexing with `[]`.
"""
function Base.getindex(pc::GradientInfo, idx::VarInfo)
  @assert haskey(pc.values, idx) "GradientInfo has no $idx."
  return pc.values[idx]
end

doc"""
    Base.setindex!(pc::GradientInfo, val, idx::VarInfo)

Make the prior values support assignment with `[]`.
"""
function Base.setindex!(pc::GradientInfo, val, idx::VarInfo)
  @assert haskey(pc.values, idx) "GradientInfo has no $idx."
  pc.values[idx] = val
end

doc"""
    Base.keys(pc::GradientInfo)

Return a key interator in the values, i.e. all the priors.
"""
function Base.keys(pc::GradientInfo)
  return keys(pc.values)
end
