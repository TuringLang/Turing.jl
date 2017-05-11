import Base.string, Base.isequal, Base.==, Base.hash
import Base.getindex, Base.setindex!, Base.push!
import Base.rand, Base.show, Base.isnan, Base.isempty

###########
# VarName #
###########
immutable VarName
  csym      ::    Symbol        # symbol generated in compilation time
  sym       ::    Symbol        # variable symbol
  indexing  ::    String        # indexing
  counter   ::    Int           # counter of same {csym, uid}
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)
hash(vn::VarName) = hash(uid(vn))

isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName)      = isequal(x, y)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{VarName}) = replace(string(map(vn -> string(vn), vns)), "String", "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

###########
# VarInfo #
###########

type VarInfo
  idcs        ::    Dict{VarName, Int}
  vns         ::    Vector{VarName}
  ranges      ::    Vector{UnitRange{Int}}
  vals        ::    Vector{Vector{Real}}
  dists       ::    Vector{Distribution}
  gids        ::    Vector{Int}
  trans       ::    Vector{Vector{Bool}}
  logp        ::    Vector{Real}
  logw        ::    Real          # NOTE: importance weight when sampling from the prior.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = begin
    vals = Vector{Vector{Real}}(); push!(vals, Vector{Real}())
    trans = Vector{Vector{Real}}(); push!(trans, Vector{Real}())
    logp = Vector{Real}(); push!(logp, zero(Real))

    new(
      Dict{VarName, Int}(),
      Vector{VarName}(),
      Vector{UnitRange{Int}}(),
      vals,
      Vector{Distribution}(),
      Vector{Int}(),
      trans, logp,
      zero(Real),
      0,
      0
    )
  end
end

typealias VarView Union{Int,UnitRange,Vector{Int},Vector{UnitRange}}

getidx(vi::VarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]

getval(vi::VarInfo, vn::VarName)       = vi.vals[end][getrange(vi, vn)]
setval!(vi::VarInfo, val, vn::VarName) = vi.vals[end][getrange(vi, vn)] = val

getval(vi::VarInfo, view::VarView)                      = vi.vals[end][view]
setval!(vi::VarInfo, val::Any, view::VarView)           = vi.vals[end][view] = val
setval!(vi::VarInfo, val::Any, view::Vector{UnitRange}) = map(v->vi.vals[end][v] = val, view)

getall(vi::VarInfo)            = vi.vals[end]
setall!(vi::VarInfo, val::Any) = vi.vals[end] = val

getsym(vi::VarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]

setgid!(vi::VarInfo, gid::Int, vn::VarName) = vi.gids[getidx(vi, vn)] = gid

istrans(vi::VarInfo, vn::VarName) = vi.trans[end][getidx(vi, vn)]
settrans!(vi::VarInfo, trans::Bool, vn::VarName) = vi.trans[end][getidx(vi, vn)] = trans

getlogp(vi::VarInfo) = vi.logp[end]
setlogp!(vi::VarInfo, logp::Real) = vi.logp[end] = logp
acclogp!(vi::VarInfo, logp::Real) = vi.logp[end] += logp

isempty(vi::VarInfo) = isempty(vi.idcs)

# X -> R for all variables associated with given sampler
link!(vi::VarInfo, spl::Sampler) = begin
  for vn in getvns(vi, spl)
    dist = getdist(vi, vn)
    setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
    settrans!(vi, true, vn)
  end
  setlogp!(vi, zero(Real))
end

# R -> X for all variables associated with given sampler
invlink!(vi::VarInfo, spl::Sampler) = begin
  for vn in getvns(vi, spl)
    dist = getdist(vi, vn)
    setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
    settrans!(vi, false, vn)
  end
  setlogp!(vi, zero(Real))
end

function cleandual!(vi::VarInfo)
  for vn in keys(vi)
    range = getrange(vi, vn)
    vi[range] = realpart(vi[range])
  end
  setlogp!(vi, realpart(getlogp(vi)))
  vi.logw = realpart(vi.logw)
end

vns(vi::VarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::VarInfo) = map(vn -> vn.sym, vns(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
Base.getindex(vi::VarInfo, vn::VarName) = begin
  @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
  dist = getdist(vi, vn)
  r = reconstruct(dist, getval(vi, vn))
  r = istrans(vi, vn) ? invlink(dist, r) : r
end

# Base.setindex!(vi::VarInfo, r, vn::VarName) = begin
#   dist = getdist(vi, vn)
#   setval!(vi, vectorize(dist, r), vn)
# end

# NOTE: vi[view] will just return what insdie vi (no transformations applied)
Base.getindex(vi::VarInfo, view::VarView)            = getval(vi, view)
Base.setindex!(vi::VarInfo, val::Any, view::VarView) = setval!(vi, val, view)

Base.getindex(vi::VarInfo, spl::Sampler)            = getval(vi, getranges(vi, spl))
Base.setindex!(vi::VarInfo, val::Any, spl::Sampler) = setval!(vi, val, getranges(vi, spl))

Base.getindex(vi::VarInfo, spl::Void)            = getall(vi)
Base.setindex!(vi::VarInfo, val::Any, spl::Void) = setall!(vi, val)

Base.keys(vi::VarInfo) = keys(vi.idcs)

Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, vn)

Base.show(io::IO, vi::VarInfo) = begin
  vi_str = """
  /=======================================================================
  | VarInfo
  |-----------------------------------------------------------------------
  | Varnames  :   $(string(vi.vns))
  | Range     :   $(vi.ranges)
  | Vals      :   $(vi.vals)
  | GIDs      :   $(vi.gids)
  | Trans?    :   $(vi.trans)
  | Logp      :   $(vi.logp)
  | Logw      :   $(vi.logw)
  | Index     :   $(vi.index)
  | #produce  :   $(vi.num_produce)
  \\=======================================================================
  """
  print(io, vi_str)
end

# Add a new entry to VarInfo
push!(vi::VarInfo, vn::VarName, r::Any, dist::Distribution, gid::Int) = begin

  @assert ~(vn in vns(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

  val = vectorize(dist, r)

  vi.idcs[vn] = length(vi.idcs) + 1
  push!(vi.vns, vn)
  l = length(vi.vals[end]); n = length(val)
  push!(vi.ranges, l+1:l+n)
  append!(vi.vals[end], val)
  push!(vi.dists, dist)
  push!(vi.gids, gid)
  push!(vi.trans[end], false)

  vi
end

# This method is use to generate a new VarName with the right count
VarName(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
  VarName(csym, sym, indexing, 1)
end

#################################
# Utility functions for VarInfo #
#################################

expand!(vi::VarInfo) = begin
  push!(vi.vals, realpart(vi.vals[end])); vi.vals[end], vi.vals[end-1] = vi.vals[end-1], vi.vals[end]
  push!(vi.trans, deepcopy(vi.trans[end]))
  push!(vi.logp, zero(Real))
end

shrink!(vi::VarInfo) = begin
  pop!(vi.vals)
  pop!(vi.trans)
  pop!(vi.logp)
end

last!(vi::VarInfo) = begin
  vi.vals = vi.vals[end:end]
  vi.trans = vi.trans[end:end]
  vi.logp = vi.logp[end:end]
end

# Get all indices of variables belonging to gid or 0
getidcs(vi::VarInfo) = getidcs(vi, nothing)
getidcs(vi::VarInfo, spl::Void) = filter(i -> vi.gids[i] == 0 || vi.gids[i] == 0, 1:length(vi.gids))
getidcs(vi::VarInfo, spl::Sampler) = begin
  # NOTE: 0b00 is the sanity flag for
  #         |\____ getidcs   (mask = 0b10)
  #         \_____ getranges (mask = 0b01)
  if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
  if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
    spl.info[:idcs]
  else
    spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
    spl.info[:idcs] = filter(i ->
      (vi.gids[i] == spl.alg.gid || vi.gids[i] == 0) && (isempty(spl.alg.space) || vi.vns[i].sym in spl.alg.space),
      1:length(vi.gids)
    )
  end
end

# Get all values of variables belonging to gid or 0
getvals(vi::VarInfo) = getvals(vi, nothing)
getvals(vi::VarInfo, spl::Union{Void, Sampler}) = map(i -> vi[vi.ranges[i]], getidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
getvns(vi::VarInfo) = getvns(vi, nothing)
getvns(vi::VarInfo, spl::Union{Void, Sampler}) = map(i -> vi.vns[i], getidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
getranges(vi::VarInfo, spl::Sampler) = begin
  if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
  if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
    spl.info[:ranges]
  else
    spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
    spl.info[:ranges] = union(map(i -> vi.ranges[i], getidcs(vi, spl))...)
  end
end

getretain(vi::VarInfo, n_retain::Int, spl::Union{Void, Sampler}) = begin
  gidcs = getidcs(vi, spl)
  UnitRange[map(i -> vi.ranges[gidcs[i]], length(gidcs):-1:(n_retain + 1))...]
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

# Check if a vn is set to NULL
isnan(vi::VarInfo, vn::VarName) = any(isnan(getval(vi, vn)))

# Sanity check for VarInfo.index
checkindex(vn::VarName, vi::VarInfo) = checkindex(vn, vi, nothing)
checkindex(vn::VarName, vi::VarInfo, spl::Union{Void, Sampler}) = begin
  vn_index = getvns(vi, spl)[vi.index]
  @assert vn_index == vn " sanity check for VarInfo.index failed: vn_index=$vn_index, vi.index=$(vi.index), vn_now=$(vn)"
end

updategid!(vi::VarInfo, vn::VarName, spl::Sampler) = begin
  if ~isempty(spl.alg.space) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
    setgid!(vi, spl.alg.gid, vn)
  end
end
