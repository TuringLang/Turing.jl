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
Base.hash(vn::VarName) = hash(uid(vn))

isequal(x::VarName, y::VarName) = hash(uid(x)) == hash(uid(y))
==(x::VarName, y::VarName)      = isequal(x, y)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{VarName}) = replace(string(map(vn -> string(vn), vns)), "String", "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

copybyindex(vn::VarName, indexing::String) = VarName(vn.csym, vn.sym, indexing, vn.counter)

###########
# VarInfo #
###########

type VarInfo
  idcs        ::    Dict{VarName,Int}
  vns         ::    Vector{VarName}
  ranges      ::    Vector{UnitRange{Int}}
  vals        ::    Vector{Vector{Real}}
  dists       ::    Vector{Distributions.Distribution}
  gids        ::    Vector{Int}
  trans       ::    Vector{Vector{Bool}}
  logp        ::    Vector{Real}
  pred        ::    Dict{Symbol,Any}
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  orders      ::    Vector{Int}   # observe statements number associated with random variables
  VarInfo() = begin
    vals = Vector{Vector{Real}}(); push!(vals, Vector{Real}())
    trans = Vector{Vector{Real}}(); push!(trans, Vector{Real}())
    logp = Vector{Real}(); push!(logp, zero(Real))
    pred = Dict{Symbol,Any}()

    new(
      Dict{VarName, Int}(),
      Vector{VarName}(),
      Vector{UnitRange{Int}}(),
      vals,
      Vector{Distributions.Distribution}(),
      Vector{Int}(),
      trans, logp,
      pred,
      0,
      Vector{Int}()
    )
  end
end

const VarView = Union{Int,UnitRange,Vector{Int},Vector{UnitRange}}

getidx(vi::VarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
getranges(vi::VarInfo, vns::Vector{VarName}) = union(map(vn -> getrange(vi, vn), vns)...)

getval(vi::VarInfo, vn::VarName)       = view(vi.vals[end], getrange(vi, vn))
setval!(vi::VarInfo, val, vn::VarName) = vi.vals[end][getrange(vi, vn)] = val

getval(vi::VarInfo, vns::Vector{VarName}) = view(vi.vals[end], getranges(vi, vns))

getval(vi::VarInfo, vview::VarView)                      = view(vi.vals[end], vview)
setval!(vi::VarInfo, val::Any, vview::VarView)           = vi.vals[end][vview] = val
setval!(vi::VarInfo, val::Any, vview::Vector{UnitRange}) = length(vview) > 0 ? (vi.vals[end][[i for arr in vview for i in arr]] = val) : nothing

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
resetlogp!(vi::VarInfo) = setlogp!(vi, zero(Real))

isempty(vi::VarInfo) = isempty(vi.idcs)

# X -> R for all variables associated with given sampler
link!(vi::VarInfo, spl::Sampler) = begin
  vns = getvns(vi, spl)
  if ~istrans(vi, vns[1])
    for vn in vns
      dist = getdist(vi, vn)
      setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
      settrans!(vi, true, vn)
    end
  else
    warn("[Turing] attempt to link a linked vi")
  end
end

# R -> X for all variables associated with given sampler
invlink!(vi::VarInfo, spl::Sampler) = begin
  vns = getvns(vi, spl)
  if istrans(vi, vns[1])
    for vn in vns
      dist = getdist(vi, vn)
      setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
      settrans!(vi, false, vn)
    end
  else
    warn("[Turing] attempt to invlink an invlinked vi")
  end
end

function cleandual!(vi::VarInfo)
  for i = 1:length(vi.vals[end])
    vi.vals[end][i] = realpart(vi.vals[end][i])
  end
  vi.logp[end] = realpart(getlogp(vi))
end

vns(vi::VarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::VarInfo) = map(vn -> vn.sym, vns(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
Base.getindex(vi::VarInfo, vn::VarName) = begin
  @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
  dist = getdist(vi, vn)
  istrans(vi, vn) ?
    invlink(dist, reconstruct(dist, getval(vi, vn))) :
    reconstruct(dist, getval(vi, vn))
end

Base.getindex(vi::VarInfo, vns::Vector{VarName}) = begin
  @assert haskey(vi, vns[1]) "[Turing] attempted to replay unexisting variables in VarInfo"
  dist = getdist(vi, vns[1])
  istrans(vi, vns[1]) ?
    invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
    reconstruct(dist, getval(vi, vns), length(vns))
end

# NOTE: vi[vview] will just return what insdie vi (no transformations applied)
Base.getindex(vi::VarInfo, vview::VarView)            = getval(vi, vview)
Base.setindex!(vi::VarInfo, val::Any, vview::VarView) = setval!(vi, val, vview)

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
  | Orders    :   $(vi.orders)
  | Logp      :   $(vi.logp)
  | #produce  :   $(vi.num_produce)
  \\=======================================================================
  """
  print(io, vi_str)
end

# Add a new entry to VarInfo
push!(vi::VarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gid::Int) = begin

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

addorder!(vi::VarInfo, vn::VarName, num_produce::Int) = begin
  push!(vi.orders, num_produce)
  vi
end

setorder!(vi::VarInfo, vn::VarName, index::Int) = begin
  if vi.orders[vi.idcs[vn]] != index
    vi.orders[vi.idcs[vn]] = index
  end
  vi
end

# This method is use to generate a new VarName with the right count
VarName(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
  VarName(csym, sym, indexing, 1)
end
VarName(vi::VarInfo, syms::Vector{Symbol}, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
    VarName(syms[1], syms[2], indexing, 1)
end

#################################
# Utility functions for VarInfo #
#################################

# expand!(vi::VarInfo) = begin
#   push!(vi.vals, realpart(vi.vals[end])); vi.vals[end], vi.vals[end-1] = vi.vals[end-1], vi.vals[end]
#   push!(vi.trans, deepcopy(vi.trans[end]))
#   push!(vi.logp, zero(Real))
# end
#
# shrink!(vi::VarInfo) = begin
#   pop!(vi.vals)
#   pop!(vi.trans)
#   pop!(vi.logp)
# end
#
# last!(vi::VarInfo) = begin
#   vi.vals = vi.vals[end:end]
#   vi.trans = vi.trans[end:end]
#   vi.logp = vi.logp[end:end]
# end

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
      (vi.gids[i] == spl.alg.gid || vi.gids[i] == 0) && (isempty(spl.alg.space) || is_inside(vi.vns[i], spl.alg.space)),
      1:length(vi.gids)
    )
  end
end

is_inside(vn::VarName, space::Set)::Bool = begin
  if vn.sym in space
    true
  else
    exprs = filter(el -> isa(el, Expr), space)
    strs = map(ex -> replace(string(ex), r"\(|\)", ""), exprs)
    vn_str = string(vn.sym) * vn.indexing
    valid = filter(str -> contains(vn_str, str), strs)
    length(valid) > 0
  end
end

# Get all values of variables belonging to gid or 0
getvals(vi::VarInfo) = getvals(vi, nothing)
getvals(vi::VarInfo, spl::Union{Void, Sampler}) = view(vi.vals[end], getidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
getvns(vi::VarInfo) = getvns(vi, nothing)
getvns(vi::VarInfo, spl::Union{Void, Sampler}) = view(vi.vns, getidcs(vi, spl))

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

getretain(vi::VarInfo, spl::Union{Void, Sampler}) = begin
  gidcs = getidcs(vi, spl)
  if vi.num_produce == 0 # called at begening of CSMC sweep for non reference particles
    UnitRange[map(i -> vi.ranges[gidcs[i]], length(gidcs):-1:1)...]
  else
    retained = [idx for idx in 1:length(vi.orders) if idx in gidcs && vi.orders[idx] > vi.num_produce]
    UnitRange[map(i -> vi.ranges[i], retained)...]
  end
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

# Check if a vn is set to NULL
isnan(vi::VarInfo, vn::VarName) = any(isnan.(getval(vi, vn)))

updategid!(vi::VarInfo, vn::VarName, spl::Sampler) = begin
  if ~isempty(spl.alg.space) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
    setgid!(vi, spl.alg.gid, vn)
  end
end
