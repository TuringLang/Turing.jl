########## VarInfo ##########

type VarInfo
  vals        ::    Dict{String, Any}
  syms        ::    Dict{String, Symbol}
  dists       ::    Dict{String, Distribution}
  logjoint    ::    Dual
  VarInfo() = new(
    Dict{String, Any}(),
    Dict{String, Symbol}(),
    Dict{String, Distribution}(),
    Dual(0)
  )
end

Base.getindex(vi::VarInfo, uid::String) = vi.vals[uid]

# The default setindex!() for VarInfo is to set values
Base.setindex!(vi::VarInfo, val, uid::String) = vi.vals[uid] = val

Base.keys(vi::VarInfo) = keys(vi.vals)

syms(vi::VarInfo) = Set(values(vi.syms))

export VarInfo, syms
