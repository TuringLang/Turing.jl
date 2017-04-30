import Base.string, Base.isequal, Base.==, Base.convert
import Base.getindex, Base.setindex!
import Base.rand, Base.show

###########
# VarName #
###########
immutable VarName
  csym      ::    Symbol        # symbol generated in compilation time
  sym       ::    Symbol        # variable symbol
  indexing  ::    String        # indexing
  counter   ::    Int           # counter of same {csym, uid}
end

typealias UID Tuple{Symbol,Symbol,String,Int}

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)

Base.string(vn::VarName) = string(uid(vn))
Base.string(uid::UID) = "{$(uid[1]),$(uid[2])$(uid[3])}:$(uid[4])"
Base.string(uids::Vector{UID}) = replace(string(map(uid -> string(uid), uids)), "String", "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol
sym(t::UID) = Symbol("$(t[2])$(t[3])")

isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName) = isequal(x, y)

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing) # the uid which is only available at compile time

# This two function is necessary because vn itself cannot be used as a key
Base.getindex(graddict::Dict, vn::VarName) = graddict[uid(vn)]
Base.setindex!(graddict::Dict, val, vn::VarName) = graddict[uid(vn)] = val

Base.convert(::Type{UID}, vn::VarName) = uid(vn)

###########
# VarInfo #
###########

type VarInfo
  idcs        ::    Dict{UID, Int}
  uids        ::    Vector{UID}
  ranges      ::    Vector{UnitRange{Int}}
  vals        ::    Vector{Real}
  dists       ::    Vector{Distribution}
  gids        ::    Vector{Int}   # group ids
  trans       ::    Vector{Bool}
  logp        ::    Real
  logw        ::    Real          # NOTE: importance weight when sampling from the prior.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{UID, Int}(),
    Vector{UID}(),
    Vector{UnitRange{Int}}(),
    Vector{Real}(),
    Vector{Distribution}(),
    Vector{Int}(),
    Vector{Bool}(),
    0.0,0.0,
    0,
    0
  )
end

getidx(vi::VarInfo, vn::VarName) = vi.idcs[uid(vn)]
getidx(vi::VarInfo, uid::UID) = vi.idcs[uid]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
getrange(vi::VarInfo, uid::UID) = vi.ranges[getidx(vi, uid)]

getval(vi::VarInfo, vn::VarName) = vi.vals[getrange(vi, vn)]
getval(vi::VarInfo, uid::UID) = vi.vals[getrange(vi, uid)]
getval(vi::VarInfo, idx::Int) = vi.vals[idx]
getval(vi::VarInfo, range::UnitRange) = vi.vals[range]

setval!(vi::VarInfo, val, vn::VarName, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getrange(vi, vn)] = val
end

setval!(vi::VarInfo, val, uid::UID, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getrange(vi, uid)] = val
end

setval!(vi::VarInfo, val, idx::Int, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[idx] = val
end

setval!(vi::VarInfo, val, range::UnitRange, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[range] = val
end

getsym(vi::VarInfo, vn::VarName) = vi.uids[getidx(vi, vn)][2]
getsym(vi::VarInfo, uid::UID)  = vi.uids[getidx(vi, uid)][2]

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
getdist(vi::VarInfo, uid::UID)  = vi.dists[getidx(vi, uid)]
setdist!(vi::VarInfo, dist, vn::VarName) = vi.dists[getidx(vi, vn)] = dist
setdist!(vi::VarInfo, dist, uid::UID)  = vi.dists[getidx(vi, uid)] = dist

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
getgid(vi::VarInfo, uid::UID)  = vi.gids[getidx(vi, uid)]
setgid!(vi::VarInfo, gid, vn::VarName) = vi.gids[getidx(vi, vn)] = gid
setgid!(vi::VarInfo, gid, uid::UID)  = vi.gids[getidx(vi, uid)] = gid

istransformed(vi::VarInfo, vn::VarName) = vi.trans[getidx(vi, vn)]
istransformed(vi::VarInfo, uid::UID)  = vi.trans[getidx(vi, uid)]
settrans!(vi::VarInfo, trans, vn::VarName) = vi.trans[getidx(vi, vn)] = trans
settrans!(vi::VarInfo, trans, uid::UID)  = vi.trans[getidx(vi, uid)] = trans

uids(vi::VarInfo) = Set(keys(vi.idcs))            # get all uids
syms(vi::VarInfo) = map(uid -> uid[2], uids(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, vn::VarName)      = getval(vi, vn)
Base.getindex(vi::VarInfo, uid::UID)       = getval(vi, uid)
Base.getindex(vi::VarInfo, idx::Int)         = getval(vi, idx)
Base.getindex(vi::VarInfo, range::UnitRange) = getval(vi, range)

Base.setindex!(vi::VarInfo, val, vn::VarName)      = setval!(vi, val, vn, true)
Base.setindex!(vi::VarInfo, val, uid::UID)       = setval!(vi, val, uid, true)
Base.setindex!(vi::VarInfo, val, idx::Int)         = setval!(vi, val, idx, true)
Base.setindex!(vi::VarInfo, val, range::UnitRange) = setval!(vi, val, range, true)

Base.keys(vi::VarInfo) = map(t -> VarName(t...), keys(vi.idcs))

Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, uid(vn))

Base.show(io::IO, vi::VarInfo) = begin
  vi_str = """
  /=======================================================================
  | VarInfo
  |-----------------------------------------------------------------------
  | UIDs      :   $(string(vi.uids))
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

#################################
# Utility functions for VarInfo #
#################################

# Get all indices of variables belonging to gid or 0
groupidcs(vi::VarInfo, gid::Int) = groupidcs(vi, gid, nothing)
groupidcs(vi::VarInfo, gid::Int, spl::Void) = filter(i -> vi.gids[i] == gid || vi.gids[i] == 0, 1:length(vi.gids))
groupidcs(vi::VarInfo, gid::Int, spl::Sampler) =
  filter(i -> (vi.gids[i] == gid || vi.gids[i] == 0) && (isempty(spl.alg.space) || vi.uids[i][2] in spl.alg.space), 1:length(vi.gids))

# Get all values of variables belonging to gid or 0
groupvals(vi::VarInfo, gid::Int) = groupvals(vi, gid, nothing)
groupvals(vi::VarInfo, gid::Int, spl::Union{Void, Sampler}) = map(i -> vi.vals[vi.ranges[i]], groupidcs(vi, gid, spl))

# Get all uids of variables belonging to gid or 0
groupuids(vi::VarInfo, gid::Int) = groupuids(vi, gid, nothing)
groupuids(vi::VarInfo, gid::Int, spl::Union{Void, Sampler}) = map(i -> vi.uids[i], groupidcs(vi, gid, spl))

retain(vi::VarInfo, gid::Int, n_retain::Int) = retain(vi, gid, n_retain, nothing)
retain(vi::VarInfo, gid::Int, n_retain::Int, spl::Union{Void, Sampler}) = begin
  gidcs = groupidcs(vi, gid, spl)

  # Set all corresponding entries to NaN
  l = length(gidcs)
  for i = l:-1:(n_retain + 1),  # for each variable (in reversed order)
      j = vi.ranges[gidcs[i]]   # for each index of variable range
    vi[j] = NaN
  end

  vi
end

# Add a new entry to VarInfo
addvar!(vi::VarInfo, vn::VarName, val, dist::Distribution) = addvar!(vi, vn, val, dist, 0)
addvar!(vi::VarInfo, vn::VarName, val, dist::Distribution, gid::Int) = begin
  @assert ~(uid(vn) in uids(vi)) "[addvar!] attempt to add an exisitng variable $(sym(vn)) ($(uid(vn))) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"
  vi.idcs[uid(vn)] = length(vi.idcs) + 1
  push!(vi.uids, uid(vn))
  l, n = length(vi.vals), length(val)
  push!(vi.ranges, l+1:l+n)
  append!(vi.vals, val)
  push!(vi.dists, dist)
  push!(vi.gids, gid)
  push!(vi.trans, false)
end

# This method is use to generate a new VarName with the right count
VarName(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
  VarName(csym, sym, indexing, 1)
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

# Sanity check for VarInfo.index
checkindex(vn::VarName, vi::VarInfo, gid::Int) = checkindex(vn, vi, gid, nothing)
checkindex(vn::VarName, vi::VarInfo, gid::Int, spl::Union{Void, Sampler}) = begin
  uid_index = groupuids(vi, gid, spl)[vi.index]
  @assert uid_index == uid(vn) "[Turing]: sanity check for VarInfo.index failed: uid_index=$uid_index, vi.index=$(vi.index), uid_now=$(uid(vn))"
end

# This method is called when sampler is missing
# NOTE: this used for initialize VarInfo, i.e. vi = model()
# NOTE: this method is also used by IS
# NOTE: this method is also used by TraceR
randr(vi::VarInfo, vn::VarName, dist::Distribution) = randr(vi, vn, dist, false)
randr(vi::VarInfo, vn::VarName, dist::Distribution, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    initvar(vi, vn, dist)
  else
    if count checkindex(vn, vi, 0, nothing) end
    replayvar(vi, vn, dist)
  end
end

# Initialize VarInfo, i.e. sampling from priors
initvar(vi::VarInfo, vn::VarName, dist::Distribution) = initvar(vi, vn, dist, 0)
initvar(vi::VarInfo, vn::VarName, dist::Distribution, gid::Int) = begin
  @assert ~haskey(vi, vn) "[Turing] attempted to initialize existing variables in VarInfo"
  r = rand(dist)
  addvar!(vi, vn, vectorize(dist, r), dist, gid)
  r
end

# Replay variables
replayvar(vi::VarInfo, vn::VarName, dist::Distribution) = begin
  @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
  if istransformed(vi, vn)  # NOTE: Implement: `vi[vn::VarName]`: (vn, vi) -> (r, lp)?
    r = invlink(dist, reconstruct(dist, vi[vn])) #  logr = log(r)
    vi.logp += logpdf(dist, r, true) # logr preserves precision of r
  else
    r = reconstruct(dist, vi[vn])
    vi.logp += logpdf(dist, r, false)
  end
  r
end

# Replay variables with group IDs updated
replayvar(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler) = begin
  if ~isempty(spl.alg.space) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
    setgid!(vi, spl.alg.group_id, vn)
  end
  replayvar(vi, vn, dist)
end

# Random with replaying
randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler) = randr(vi, vn, dist, spl, false)
randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = initvar(vi, vn, dist, spl.alg.group_id)
  elseif isnan(vi[vn][1])
    r = rand(dist)
    vi[vn] = vectorize(dist, r)
  else
    if count checkindex(vn, vi, spl.alg.group_id, spl) end
    r = replayvar(vi, vn, dist, spl)
  end
  r
end
