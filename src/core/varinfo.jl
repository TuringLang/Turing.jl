import Base.string, Base.isequal, Base.==

########## VarName ##########

immutable VarName
  csym      ::    Symbol        # symbol generated in compilation time
  sym       ::    Symbol        # variable symbol
  indexing  ::    String        # indexing
  counter   ::    Int           # counter of same {csym, uid}
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
string(vn::VarName) = uid(vn)
isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName) = uid(x) == uid(y)

cuid(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}" # the uid which is only available at compile time

########## VarInfo ##########

type VarInfo
  idcs        ::    Dict{String, Int}
  vals        ::    Vector{Any}
  syms        ::    Vector{Symbol}
  dists       ::    Vector{Distribution}
  logjoint    ::    Dual

  names       ::    Vector{String}
  tsyms       ::    Vector{Symbol}
  randomness  ::    Vector{Any}   # elem t is the randomness created by the t’th assume call.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{String, Int}(),
    Vector{Any}(),
    Vector{Symbol}(),
    Vector{Distribution}(),
    Dual(0.0),
    Vector{String}(),
    Vector{Symbol}(),
    Vector{Any}(),
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

syms(vi::VarInfo) = union(Set(tsyms), Set(syms))

function randr(vi::VarInfo, name::String, sym::Symbol, distr::Distribution)
  vi.index += 1
  local r
  if vi.index <= length(vi.randomness)
    r = vi.randomness[vi.index]
  else # sample, record
    @assert ~(name in vi.names) "[randr(trace)] attempt to generate an exisitng variable $name to $(vi)"
    r = Distributions.rand(distr)
    push!(vi.randomness, r)
    push!(vi.names, name)
    push!(vi.tsyms, sym)
  end
  return r
end
