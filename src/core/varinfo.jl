########## VarInfo ##########

type VarInfo
  idcs        ::    Dict{String, Int}
  vals        ::    Vector{Any}
  syms        ::    Vector{Symbol}
  dists       ::    Vector{Distribution}
  logjoint    ::    Dual
  VarInfo() = new(
    Dict{String, Int}(),
    Vector{Any}(),
    Vector{Symbol}(),
    Vector{Distribution}(),
    Dual(0.0)
  )
end

function mapuid(vi::VarInfo, uid::String)
  if haskey(vi.idcs, uid)
    vi.idcs[uid]
  else
    vi.idcs[uid] = length(vi.idcs) + 1
  end
end

getsym(vi::VarInfo, uid::String) = vi.syms[mapuid(vi, uid)]
function setsym!(vi::VarInfo, sym, uid::String)
  idx = mapuid(vi, uid)
  if length(vi.syms) < idx
    push!(vi.syms, sym)
  else
    vi.syms[idx] = sym
  end
end

getdist(vi::VarInfo, uid::String) = vi.dists[mapuid(vi, uid)]
function setdist!(vi::VarInfo, dist, uid::String)
  idx = mapuid(vi, uid)
  if length(vi.dists) < idx
    push!(vi.dists, dist)
  else
    vi.dists[idx] = dist
  end
end

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, uid::String) = vi.vals[mapuid(vi, uid)]
function Base.setindex!(vi::VarInfo, val, uid::String)
  idx = mapuid(vi, uid)
  if length(vi.vals) < idx
    push!(vi.vals, val)
  else
    vi.vals[idx] = val
  end
end

Base.haskey(vi::VarInfo, uid::String) = haskey(vi.idcs, uid)
Base.keys(vi::VarInfo) = keys(vi.idcs)
