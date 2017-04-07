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
  vals        ::    Vector{Any}
  dists       ::    Vector{Distribution}
  gids        ::    Vector{Int}   # group ids
  istrans     ::    Vector{Bool}  # is transformed?
  logjoint    ::    Dual
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{Tuple, Int}(),
    Vector{Tuple}(),
    Vector{Any}(),
    Vector{Distribution}(),
    Vector{Int}(),
    Vector{Bool}(),
    Dual(0.0),
    0,
    0
  )
end

Base.show(io::IO, vi::VarInfo) = begin
  println(vi.idcs)
  print("$(vi.uids)\n$(vi.vals)\n$(vi.gids)\n$(vi.istrans)\n")
  print("$(vi.logjoint), $(vi.index), $(vi.num_produce)")
end

getidx(vi::VarInfo, vn::VarName) = vi.idcs[uid(vn)]
getidx(vi::VarInfo, uid::Tuple) = vi.idcs[uid]

getval(vi::VarInfo, vn::VarName) = vi.vals[getidx(vi, vn)]
getval(vi::VarInfo, uid::Tuple) = vi.vals[getidx(vi, uid)]

setval!(vi::VarInfo, val, vn::VarName, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getidx(vi, vn)] = val
end

setval!(vi::VarInfo, val, uid::Tuple, overwrite=false) = begin
  if ~overwrite
    warn("[setval!] you are overwritting values in VarInfo without setting overwrite flag to be true")
  end
  vi.vals[getidx(vi, uid)] = val
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

istrans(vi::VarInfo, vn::VarName) = vi.istrans[getidx(vi, vn)]
istrans(vi::VarInfo, uid::Tuple) = vi.istrans[getidx(vi, uid)]

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
groupvals(vi::VarInfo, gid::Int, spl=nothing) = map(i -> vi.vals[i], groupidcs(vi, gid, spl))

# Get all uids of variables belonging to gid or 0
groupuids(vi::VarInfo, gid::Int, spl=nothing) = map(i -> vi.uids[i], groupidcs(vi, gid, spl))

retain(vi::VarInfo, gid::Int, n_retain) = begin
  # NOTE: the sanity check below is commented because Void
  #       and standalone samplers uses gid = 0
  # @assert ~(gid == 0) "[retain] wrong use of retain: gid = 0"

  # Get all indices of variables belonging to gid
  gidcs = filter(i -> vi.gids[i] == gid, 1:length(vi.gids))
  l = length(gidcs)

  # Remove corresponding entries
  for i = l:-1:(n_retain + 1)
    delete!(vi.idcs, vi.uids[gidcs[i]])
    splice!(vi.uids, gidcs[i])
    splice!(vi.vals, gidcs[i])
    splice!(vi.dists, gidcs[i])
    splice!(vi.gids, gidcs[i])
    splice!(vi.istrans, gidcs[i])
  end

  # Rebuild index dictionary
  for i = 1:length(vi.uids)
    vi.idcs[vi.uids[i]] = i
  end

  vi
end

# Add a new entry to VarInfo
addvar!(vi::VarInfo, vn::VarName, val, dist::Distribution, gid=0, istrans=false) = begin
  @assert ~(uid(vn) in uids(vi)) "[addvar!] attempt to add an exisitng variable $(sym(vn)) ($(uid(vn))) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid, istrans=$istrans"
  vi.idcs[uid(vn)] = length(vi.idcs) + 1
  push!(vi.uids, uid(vn))
  push!(vi.vals, val)
  push!(vi.dists, dist)
  push!(vi.gids, gid)
  push!(vi.istrans, istrans)
end

# This method is use to generate a new VarName with the right count
nextvn(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
  VarName(csym, sym, indexing, 1)
end

# Main behaviour control of rand() depending on sampler type and if sampler inside
rand(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, inside=true) = begin
  local count, trans

  if isa(spl, HMCSampler{HMC})
    count, trans = false, true
  elseif isa(spl, ParticleSampler{PG})
    count, trans = true, false
  else
    error("[rand]: unsupported sampler: $spl")
  end

  if inside
    randr(vi, vn, dist, spl.alg.group_id, trans, spl, count)
  else
    randr(vi, vn, dist, 0, ~trans, nothing, false)
  end
end

# This method is called when sampler is Void
rand(vi::VarInfo, vn::VarName, dist::Distribution) = begin
  randr(vi, vn, dist, 0, true, nothing, false)
end

# Random with replaying
randr(vi::VarInfo, vn::VarName, dist::Distribution, gid=0, trans=false, spl=nothing, count=false) = begin
  vi.index = count ? vi.index + 1 : vi.index
  local r
  if ~haskey(vi, vn)
    r = rand(dist)
    if trans
      addvar!(vi, vn, vectorize(dist, link(dist, r)), dist, gid, trans)
    else
      addvar!(vi, vn, r, dist, gid, trans)
    end
    r
  else
    if ~(spl == nothing || isempty(spl.alg.space)) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
      setgid!(vi, gid, vn)
    end
    if trans
      dist = getdist(vi, vn)
      r = invlink(dist, reconstruct(dist, vi[vn]))
    else
      r = vi[vn]
    end
    if count  # sanity check for VarInfo.index
      uid_replay = groupuids(vi, gid, spl)[vi.index]
      @assert uid_replay == uid(vn) "[randr] variable replayed doesn't match counting index"
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
