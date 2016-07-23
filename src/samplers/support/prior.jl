import Base: string
export Prior, PriorSym, PriorArr

abstract Prior

immutable PriorSym <: Prior
  Sym       ::    Symbol
  function PriorSym(Sym)
    new(Sym)
  end
end

immutable PriorArr <: Prior
  ArrExpr   ::    Expr
  IdxSym    ::    Symbol
  IdxVal    ::    Any
  function PriorArr(ArrExpr, IdxSym, IdxVal)
    new(ArrExpr, IdxSym, IdxVal)
  end
end

function string(p::Prior)
  if isa(p, PriorSym)
    return string(p.Sym)
  end
  if isa(p, PriorArr)
    arrexpr = p.ArrExpr
    if isa(arrexpr.args[2], Symbol)
      @assert arrexpr.args[2] == p.IdxSym
      arrexpr.args[2] = p.IdxVal
    end
    return string(arrexpr)
  end
end
