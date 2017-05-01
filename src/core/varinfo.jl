import Base.string, Base.isequal, Base.==, Base.hash
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

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{VarName}) = replace(string(map(vn -> string(vn), vns)), "String", "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName) = isequal(x, y)
hash(vn::VarName) =  hash(hash(vn.csym) + hash(vn.sym) + hash(vn.indexing) + hash(vn.counter))

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing) # the uid which is only available at compile time

###########
# VarInfo #
###########

type VarInfo
  idcs        ::    Dict{VarName, Int}
  vns         ::    Vector{VarName}
  ranges      ::    Vector{UnitRange{Int}}
  vals        ::    Vector{Vector{Real}}
  dists       ::    Vector{Distribution}
  gids        ::    Vector{Int}   # group ids
  trans       ::    Vector{Bool}
  logp        ::    Real
  logw        ::    Real          # NOTE: importance weight when sampling from the prior.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{VarName, Int}(),
    Vector{VarName}(),
    Vector{UnitRange{Int}}(),
    Vector{Vector{Real}}(),
    Vector{Distribution}(),
    Vector{Int}(),
    Vector{Bool}(),
    0.0,0.0,
    0,
    0
  )
end

getidx(vi::VarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]

getval(vi::VarInfo, vn::VarName)        = vi.vals[end][getrange(vi, vn)]
getval(vi::VarInfo, idx::Int)           = vi.vals[end][idx]
getval(vi::VarInfo, range::UnitRange)   = vi.vals[end][range]
getval(vi::VarInfo, range::Vector{Int}) = vi.vals[end][range]

setval!(vi::VarInfo, val, vn::VarName, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[end][getrange(vi, vn)] = val
end

setval!(vi::VarInfo, val, idx::Int, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[end][idx] = val
end

setval!(vi::VarInfo, val, range::UnitRange, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[end][range] = val
end

setval!(vi::VarInfo, val, range::Vector{Int}, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[end][range] = val
end

getsym(vi::VarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
setdist!(vi::VarInfo, dist, vn::VarName) = vi.dists[getidx(vi, vn)] = dist

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
setgid!(vi::VarInfo, gid, vn::VarName) = vi.gids[getidx(vi, vn)] = gid

istransformed(vi::VarInfo, vn::VarName) = vi.trans[getidx(vi, vn)]
settrans!(vi::VarInfo, trans, vn::VarName) = vi.trans[getidx(vi, vn)] = trans

vns(vi::VarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::VarInfo) = map(vn -> vn.sym, vns(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, vn::VarName)        = getval(vi, vn)
Base.getindex(vi::VarInfo, idx::Int)           = getval(vi, idx)
Base.getindex(vi::VarInfo, range::UnitRange)   = getval(vi, range)
Base.getindex(vi::VarInfo, range::Vector{Int}) = getval(vi, range)

Base.setindex!(vi::VarInfo, val, vn::VarName)        = setval!(vi, val, vn, true)
Base.setindex!(vi::VarInfo, val, idx::Int)           = setval!(vi, val, idx, true)
Base.setindex!(vi::VarInfo, val, range::UnitRange)   = setval!(vi, val, range, true)
Base.setindex!(vi::VarInfo, val, range::Vector{Int}) = setval!(vi, val, range, true)

Base.keys(vi::VarInfo) = keys(vi.idcs)

Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, vn)

Base.show(io::IO, vi::VarInfo) = begin
  vi_str = """
  /=======================================================================
  | VarInfo
  |-----------------------------------------------------------------------
  | Varnames  :   $(string(vi.vns))
  | Range     :   $(vi.ranges)
  | Vals      :   $(vi.vals[end])
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

duplicate!(vi::VarInfo) = push!(vi.vals, vi.vals[end])
keeplast!(vi::VarInfo) = splice!(vi.vals, 1:length(vi.vals)-1)

# Get all indices of variables belonging to gid or 0
groupidcs(vi::VarInfo, gid::Int) = groupidcs(vi, gid, nothing)
groupidcs(vi::VarInfo, gid::Int, spl::Void) = filter(i -> vi.gids[i] == gid || vi.gids[i] == 0, 1:length(vi.gids))
groupidcs(vi::VarInfo, gid::Int, spl::Sampler) =
  filter(i -> (vi.gids[i] == gid || vi.gids[i] == 0) && (isempty(spl.alg.space) || vi.vns[i].sym in spl.alg.space), 1:length(vi.gids))

# Get all values of variables belonging to gid or 0
groupvals(vi::VarInfo, gid::Int) = groupvals(vi, gid, nothing)
groupvals(vi::VarInfo, gid::Int, spl::Union{Void, Sampler}) = map(i -> vi.vals[end][vi.ranges[i]], groupidcs(vi, gid, spl))

# Get all vns of variables belonging to gid or 0
groupvns(vi::VarInfo, gid::Int) = groupvns(vi, gid, nothing)
groupvns(vi::VarInfo, gid::Int, spl::Union{Void, Sampler}) = map(i -> vi.vns[i], groupidcs(vi, gid, spl))

# Get all vns of variables belonging to gid or 0
getranges(vi::VarInfo, spl::Sampler) = union(map(i -> vi.ranges[i], groupidcs(vi, spl.alg.group_id, spl))...)

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
  @assert ~(vn in vns(vi)) "[addvar!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"
  if isempty(vi.vals) push!(vi.vals, Vector{Real}()) end
  vi.idcs[vn] = length(vi.idcs) + 1
  push!(vi.vns, vn)
  l, n = length(vi.vals[end]), length(val)
  push!(vi.ranges, l+1:l+n)
  append!(vi.vals[end], val)
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
  vn_index = groupvns(vi, gid, spl)[vi.index]
  @assert vn_index == vn "[Turing]: sanity check for VarInfo.index failed: vn_index=$vn_index, vi.index=$(vi.index), vn_now=$(vn)"
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
