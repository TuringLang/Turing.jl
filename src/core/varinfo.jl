import Base.string, Base.isequal, Base.==, Base.convert
import Base.getindex, Base.setindex!
import Base.rand

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
  vals        ::    Vector{Any}
  syms        ::    Vector{Symbol}
  dists       ::    Vector{Distribution}
  logjoint    ::    Dual

  names       ::    Vector{Tuple}
  tsyms       ::    Vector{Symbol}
  randomness  ::    Vector{Any}   # elem t is the randomness created by the t’th assume call.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{Tuple, Int}(),
    Vector{Any}(),
    Vector{Symbol}(),
    Vector{Distribution}(),
    Dual(0.0),
    Vector{Tuple}(),
    Vector{Symbol}(),
    Vector{Any}(),
    0,
    0
  )
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

getsym(vi::VarInfo, vn::VarName) = vi.syms[getidx(vi, vn)]
getsym(vi::VarInfo, uid::Tuple) = vi.syms[getidx(vi, uid)]
setsym!(vi::VarInfo, sym, vn::VarName) = vi.syms[getidx(vi, vn)] = sym
setsym!(vi::VarInfo, sym, uid::Tuple) = vi.syms[getidx(vi, uid)] = sym

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
getdist(vi::VarInfo, uid::Tuple) = vi.dists[getidx(vi, uid)]
setdist!(vi::VarInfo, dist, vn::VarName) = vi.dists[getidx(vi, vn)] = dist
setdist!(vi::VarInfo, dist, uid::Tuple) = vi.dists[getidx(vi, uid)] = dist

# The default getindex & setindex!() for get & set values
Base.getindex(vi::VarInfo, vn::VarName) = getval(vi, vn)
Base.getindex(vi::VarInfo, uid::Tuple) = getval(vi, uid)
Base.setindex!(vi::VarInfo, val, vn::VarName) = setval!(vi, val, vn, true)
Base.setindex!(vi::VarInfo, val, uid::Tuple) = setval!(vi, val, uid, true)

addvar!(vi::VarInfo, vn::VarName, val, dist::Distribution) = begin
  @assert ~haskey(vi, vn)
  vi.idcs[uid(vn)] = length(vi.idcs) + 1
  push!(vi.vals, val)
  push!(vi.syms, vn.sym)
  push!(vi.dists, dist)
end

syms(vi::VarInfo) = union(Set(vi.tsyms), Set(vi.syms))
uids(vi::VarInfo) = union(Set(keys(vi.idcs)), Set(vi.names))

# TODO: change below after randr() is unified
Base.keys(vi::VarInfo) = map(t -> VarName(t...), keys(vi.idcs))
Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, uid(vn))

nextvn(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when VarInfo internal structure is updated
  VarName(csym, sym, indexing, 1)
end

# TODO: below should be updated when the field group is add to InferenceAlgorithm
rand(vi::VarInfo, vn::VarName, dist::Distribution, method::Symbol) = begin
  if method == :byname
    randrn(vi, vn, dist)
  elseif method == :bycounter
    randrc(vi, vn, dist)
  else
    error("[rand]: unsupported randomness replaying method: $method")
  end
end

# Random with replaying by name
randrn(vi::VarInfo, vn::VarName, dist::Distribution) = begin
  local r
  if ~haskey(vi, vn)
    dprintln(2, "sampling prior...")
    r = rand(dist)
    val = vectorize(dist, link(dist, r))      # X -> R and vectorize
    addvar!(vi, vn, val, dist)
  else
    dprintln(2, "fetching vals...")
    val = vi[vn]
    r = invlink(dist, reconstruct(dist, val)) # R -> X and reconstruct
  end
  r
end

# Random with replaying by counter
function randrc(vi::VarInfo, vn::VarName, dist::Distribution)
  vi.index += 1
  local r
  if vi.index <= length(vi.randomness)
    r = vi.randomness[vi.index]
  else # sample, record
    @assert ~(vn in vi.names) "[randr(trace)] attempt to generate an exisitng variable $name to $(vi)"
    r = Distributions.rand(dist)
    push!(vi.randomness, r)
    push!(vi.names, vn)
    push!(vi.tsyms, vn.sym)
  end
  r
end

# Randome with force overwriting by counter
function randoc(vi::VarInfo, vn::VarName, dist::Distribution)
  vi.index += 1
  r = Distributions.rand(dist)
  if vi.index <= length(vi.randomness)
    vi.randomness[vi.index] = r
  else # sample, record
    @assert ~(vn in vi.names) "[randr(trace)] attempt to generate an exisitng variable $name to $(vi)"
    push!(vi.randomness, r)
    push!(vi.names, vn)
    push!(vi.tsyms, vn.sym)
  end
  r
end
