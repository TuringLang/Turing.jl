########## VarInfo ##########

type VarInfo
  vals        ::    Dict{String, Any}
  syms        ::    Dict{String, Symbol}
  dists       ::    Dict{String, Distribution}
  logjoint    ::    Dual
  randomness :: Array{Any, 1}    # elem t is the randomness created by the t’th assume call.
  index :: Int                   # index of current randomness
  num_produce :: Int             # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{String, Any}(),
    Dict{String, Symbol}(),
    Dict{String, Distribution}(),
    Dual(0),
    Array{Any,1}(),
    0,
    0
  )
end

Base.getindex(vi::VarInfo, uid::String) = vi.vals[uid]

# The default setindex!() for VarInfo is to set values
Base.setindex!(vi::VarInfo, val, uid::String) = vi.vals[uid] = val

Base.keys(vi::VarInfo) = keys(vi.vals)

syms(vi::VarInfo) = Set(values(vi.syms))

export VarInfo, syms
