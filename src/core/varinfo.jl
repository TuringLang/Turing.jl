import Base.string, Base.isequal, Base.==, Base.convert
import Base.getindex, Base.setindex!
import Base.rand, Base.show

########## VarName ##########

immutable VarName
  csym      ::    Symbol        # symbol generated in compilation time
  sym       ::    Symbol        # variable symbol
  indexing  ::    String        # indexing
  counter   ::    Int           # counter of same {csym, uid}
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)
string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol
sym(t::Tuple{Symbol,Symbol,String,Int}) = Symbol("$(t[2])$(t[3])")

isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName) = isequal(x, y)

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing) # the uid which is only available at compile time

# This two function is necessary because vn itself cannot be used as a key
Base.getindex(graddict::Dict, vn::VarName) = graddict[uid(vn)]
Base.setindex!(graddict::Dict, val, vn::VarName) = graddict[uid(vn)] = val

Base.convert(::Type{Tuple}, vn::VarName) = uid(vn)

########## VarInfo ##########

type VarInfo
  idcs        ::    Dict{Tuple, Int}
  uids        ::    Vector{Tuple}
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
    Dict{Tuple, Int}(),
    Vector{Tuple}(),
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

Base.show(io::IO, vi::VarInfo) = begin
  println(vi.idcs)
  print("$(vi.uids)\n$(vi.ranges)\n$(vi.vals)\n$(vi.gids)\n$(vi.trans)\n")
  print("$(vi.logp), $(vi.index), $(vi.num_produce)")
end

getidx(vi::VarInfo, vn::VarName) = vi.idcs[uid(vn)]
getidx(vi::VarInfo, uid::Tuple) = vi.idcs[uid]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
getrange(vi::VarInfo, uid::Tuple) = vi.ranges[getidx(vi, uid)]

getval(vi::VarInfo, vn::VarName) = vi.vals[getrange(vi, vn)]
getval(vi::VarInfo, uid::Tuple) = vi.vals[getrange(vi, uid)]

setval!(vi::VarInfo, val, vn::VarName, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getrange(vi, vn)] = val
end

setval!(vi::VarInfo, val, uid::Tuple, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getrange(vi, uid)] = val
end

getsym(vi::VarInfo, vn::VarName) = vi.uids[getidx(vi, vn)][2]
getsym(vi::VarInfo, uid::Tuple) = vi.uids[getidx(vi, uid)][2]

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
getdist(vi::VarInfo, uid::Tuple) = vi.dists[getidx(vi, uid)]
setdist!(vi::VarInfo, dist, vn::VarName) = vi.dists[getidx(vi, vn)] = dist
setdist!(vi::VarInfo, dist, uid::Tuple) = vi.dists[getidx(vi, uid)] = dist

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
getgid(vi::VarInfo, uid::Tuple) = vi.gids[getidx(vi, uid)]
setgid!(vi::VarInfo, gid, vn::VarName) = vi.gids[getidx(vi, vn)] = gid
setgid!(vi::VarInfo, gid, uid::Tuple) = vi.gids[getidx(vi, uid)] = gid

istransformed(vi::VarInfo, vn::VarName) = vi.trans[getidx(vi, vn)]
istransformed(vi::VarInfo, uid::Tuple) = vi.trans[getidx(vi, uid)]
settrans!(vi::VarInfo, trans, vn::VarName) = vi.trans[getidx(vi, vn)] = trans
settrans!(vi::VarInfo, trans, uid::Tuple) = vi.trans[getidx(vi, uid)] = trans

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, vn::VarName) = getval(vi, vn)
Base.getindex(vi::VarInfo, uid::Tuple) = getval(vi, uid)
Base.setindex!(vi::VarInfo, val, vn::VarName) = setval!(vi, val, vn, true)
Base.setindex!(vi::VarInfo, val, uid::Tuple) = setval!(vi, val, uid, true)

uids(vi::VarInfo) = Set(keys(vi.idcs))            # get all uids
syms(vi::VarInfo) = map(uid -> uid[2], uids(vi))  # get all symbols

Base.keys(vi::VarInfo) = map(t -> VarName(t...), keys(vi.idcs))
Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, uid(vn))

# Get all indices of variables belonging to gid or 0
groupidcs(vi::VarInfo, gid::Int, spl=nothing) = begin
  if spl == nothing || isempty(spl.alg.space)
    filter(i -> vi.gids[i] == gid || vi.gids[i] == 0, 1:length(vi.gids))
  else
    filter(i -> (vi.gids[i] == gid || vi.gids[i] == 0) && (vi.uids[i][2] in spl.alg.space), 1:length(vi.gids))
  end
end

# Get all values of variables belonging to gid or 0
groupvals(vi::VarInfo, gid::Int, spl=nothing) = map(i -> vi.vals[vi.ranges[i]], groupidcs(vi, gid, spl))

# Get all uids of variables belonging to gid or 0
groupuids(vi::VarInfo, gid::Int, spl=nothing) = map(i -> vi.uids[i], groupidcs(vi, gid, spl))

retain(vi::VarInfo, gid::Int, n_retain, spl=nothing) = begin
  # NOTE: the sanity check below is commented because Void
  #       and standalone samplers uses gid = 0
  # @assert ~(gid == 0) "[retain] wrong use of retain: gid = 0"

  # Get all indices of variables belonging to gid
  gidcs = groupidcs(vi, gid, spl)
  l = length(gidcs)

  # Remove corresponding entries
  for i = l:-1:(n_retain + 1)
    for r_i in vi.ranges[gidcs[i]]
      vi.vals[r_i] = NaN
    end
    # delete!(vi.idcs, vi.uids[gidcs[i]])
    # splice!(vi.uids, gidcs[i])
    # splice!(vi.vals, gidcs[i])
    # splice!(vi.dists, gidcs[i])
    # splice!(vi.gids, gidcs[i])
    # splice!(vi.trans, gidcs[i])
  end

  # Rebuild index dictionary
  # for i = 1:length(vi.uids)
  #   vi.idcs[vi.uids[i]] = i
  # end
  vi
end

# Add a new entry to VarInfo
addvar!(vi::VarInfo, vn::VarName, val, dist::Distribution, gid=0) = begin
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

# This method is called when sampler is of type Void
rand(vi::VarInfo, vn::VarName, dist::Distribution) = randr(vi, vn, dist)

# Random with replaying
randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count=false) = begin
  gid = spl.alg.group_id
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    # Always store vector inside VarInfo
    addvar!(vi, vn, vectorize(dist, r), dist, gid)
  elseif isnan(vi[vn][1])
    r = rand(dist)
    vi[vn] = vectorize(dist, r)
  else
    if count  # sanity check for VarInfo.index
      uid_replay = groupuids(vi, gid, spl)[vi.index]
      @assert uid_replay == uid(vn) "[Turing]: `randr` variable replayed doesn't match counting index.\n
                    \t Details: uid_replay=$uid_replay, vi.index=$(vi.index), uid(vn)=$(uid(vn))"
    end
    if ~isempty(spl.alg.space) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
      setgid!(vi, gid, vn)
    end
    if istransformed(vi, vn)  # NOTE: Implement: `vi[vn::VarName]`: (vn, vi) -> (r, lp)?
      if isa(dist, SimplexDistribution)
        r = invlink(dist, reconstruct(dist, vi[vn])) #  logr = log(r)
        vi.logp += logpdf(dist, r, true) # logr preserves precision of r
      else
        r = invlink(dist, reconstruct(dist, vi[vn])) #  logr = log(r)
        vi.logp += logpdf(dist, r, true) # logr preserves precision of r
      end
    else
      r = reconstruct(dist, vi[vn])
      vi.logp += logpdf(dist, r, false)
    end
  end
  r
end

# Simple `randr` for simulating from the prior
randr(vi::VarInfo, vn::VarName, dist::Distribution, count = false) = begin
  gid = 0 # Default gid without samplers
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    r = rand(dist)
    # Always store vector inside VarInfo
    addvar!(vi, vn, vectorize(dist, r), dist, gid)
  else
    if count  # sanity check for VarInfo.index
      uid_replay = groupuids(vi, gid, spl)[vi.index]
      @assert uid_replay == uid(vn) "[Turing]: `randr` variable replayed doesn't match counting index.\n
                    \t Details: uid_replay=$uid_replay, vi.index=$(vi.index), uid(vn)=$(uid(vn))"
    end
    if istransformed(vi, vn)  # NOTE: Implement: `vi[vn::VarName]`: (vn, vi) -> (r, lp)?
      r = invlink(dist, reconstruct(dist, vi[vn])) #  logr = log(r)
      vi.logp += logpdf(dist, r, true) # logr preserves precision of r
    else
      r = reconstruct(dist, vi[vn])
      vi.logp += logpdf(dist, r, false)
    end
  end
  r
end

# Randome with force overwriting by counter
function randoc(vi::VarInfo, vn::VarName, dist::Distribution, gid=0)
  vi.index += 1
  r = Distributions.rand(dist)
  vals = groupvals(vi, 0)
  if vi.index <= length(vi.vals)
    vals[vi.index] = r
  else # sample, record
    @assert ~(uid(vn) in groupuids(vi, gid)) "[randoc] attempt to generate an exisitng variable $(sym(vn)) to $vi"
    addvar!(vi, vn, r, dist, 0)
  end
  r
end
