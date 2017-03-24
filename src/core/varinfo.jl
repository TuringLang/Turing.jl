########## VarInfo ##########

type VarInfo
  idcs        ::    Dict{String, Int}
  vals        ::    Vector{Any}
  syms        ::    Vector{Symbol}
  dists       ::    Vector{Distribution}
  logjoint    ::    Dual
  randomness  ::    Vector{Any}   # elem t is the randomness created by the t’th assume call.
  index :: Int                    # index of current randomness
  num_produce :: Int              # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{String, Int}(),
    Vector{Any}(),
    Vector{Symbol}(),
    Vector{Distribution}(),
    Dual(0.0),
    Array{Any,1}(),
    0,
    0
  )
end

getidx(vi::VarInfo, uid::String) = vi.idcs[uid]

getval(vi::VarInfo, uid::String) = vi.vals[getidx(vi, uid)]
setval!(vi::VarInfo, val, uid::String) = vi.vals[getidx(vi, uid)] = val

getsym(vi::VarInfo, uid::String) = vi.syms[getidx(vi, uid)]
setsym!(vi::VarInfo, sym, uid::String) = vi.syms[getidx(vi, uid)] = sym

getdist(vi::VarInfo, uid::String) = vi.dists[getidx(vi, uid)]
setdist!(vi::VarInfo, dist, uid::String) = vi.dists[getidx(vi, uid)] = dist

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, uid::String) = getval(vi, uid)
Base.setindex!(vi::VarInfo, val, uid::String) = setval!(vi, val, uid)

addvar!(vi::VarInfo, uid::String, val, sym::Symbol, dist::Distribution) = begin
  @assert ~haskey(vi, uid)
  vi.idcs[uid] = length(vi.idcs) + 1
  push!(vi.vals, val)
  push!(vi.syms, sym)
  push!(vi.dists, dist)
end

Base.haskey(vi::VarInfo, uid::String) = haskey(vi.idcs, uid)
Base.keys(vi::VarInfo) = keys(vi.idcs)

sync(vi::VarInfo, uid::String,r) = vi.randomness[getidx(vi, uid)] = r
sync(vi::VarInfo, uids::Vector{String}) = begin
  for uid = uids
    idx = getidx(vi, uid)
    dist = getdist(vi, uid)
    val = getval(vi, uid)
    vi.randomness[idx] = invlink(dist, reconstruct(dist, val))
  end
end
