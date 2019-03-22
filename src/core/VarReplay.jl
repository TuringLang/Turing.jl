module VarReplay

using ...Turing: Turing, CACHERESET, CACHEIDCS, CACHERANGES, Model,
    AbstractSampler, Sampler, SampleFromPrior,
    Selector
using ...Utilities: vectorize, reconstruct, reconstruct!
using Bijectors: SimplexDistribution
using Distributions

import Base: string, isequal, ==, hash, getindex, setindex!, push!, show, isempty
import Turing: link, invlink

export  VarName, 
        VarInfo, 
        uid, 
        sym, 
        getlogp, 
        set_retained_vns_del_by_spl!, 
        resetlogp!, 
        is_flagged, 
        unset_flag!, 
        setgid!, 
        copybyindex, 
        setorder!, 
        updategid!, 
        acclogp!, 
        istrans, 
        link!, 
        invlink!, 
        setlogp!, 
        getranges, 
        getrange, 
        getvns, 
        getval

###########
# VarName #
###########
struct VarName
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
Base.string(vns::Vector{VarName}) = replace(string(map(vn -> string(vn), vns)), "String" => "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

copybyindex(vn::VarName, indexing::String) = VarName(vn.csym, vn.sym, indexing, vn.counter)

###########
# VarInfo #
###########

mutable struct VarInfo
    idcs        ::    Dict{VarName,Int}
    vns         ::    Vector{VarName}
    ranges      ::    Vector{UnitRange{Int}}
    vals        ::    Vector{Real}
    rvs         ::    Dict{Union{VarName,Vector{VarName}},Any}
    dists       ::    Vector{Distributions.Distribution}
    gids        ::    Vector{Set{Selector}}
    logp        ::    Real
    num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
    orders      ::    Vector{Int}   # observe statements number associated with random variables
    flags       ::    Dict{String,Vector{Bool}}

    function VarInfo()
        vals  = Vector{Real}()
        rvs   = Dict{Union{VarName,Vector{VarName}},Any}()
        logp  = zero(Real)
        flags = Dict{String,Vector{Bool}}()
        flags["del"] = Vector{Bool}()
        flags["trans"] = Vector{Bool}()

        new(
            Dict{VarName, Int}(),
            Vector{VarName}(),
            Vector{UnitRange{Int}}(),
            vals,
            rvs,
            Vector{Distributions.Distribution}(),
            Vector{Int}(),
            logp,
            0,
            Vector{Int}(),
            flags
        )
    end
end

@generated function Turing.runmodel!(model::Model, vi::VarInfo, spl::AbstractSampler)
    expr_eval_num = spl <: Sampler ?
        :(if :eval_num ∈ keys(spl.info) spl.info[:eval_num] += 1 end) : :()
    return quote
        setlogp!(vi, zero(Real))
        $(expr_eval_num)
        model(vi, spl)
        return vi
    end
end
Turing.runmodel!(model::Model, vi::VarInfo) = Turing.runmodel!(model, vi, SampleFromPrior())

const VarView = Union{Int,UnitRange,Vector{Int},Vector{UnitRange}}

getidx(vi::VarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
getranges(vi::VarInfo, vns::Vector{VarName}) = union(map(vn -> getrange(vi, vn), vns)...)

getval(vi::VarInfo, vn::VarName)       = view(vi.vals, getrange(vi, vn))
setval!(vi::VarInfo, val, vn::VarName) = vi.vals[getrange(vi, vn)] = val

getval(vi::VarInfo, vns::Vector{VarName}) = view(vi.vals, getranges(vi, vns))

getval(vi::VarInfo, vview::VarView)                      = view(vi.vals, vview)
setval!(vi::VarInfo, val::Any, vview::VarView)           = vi.vals[vview] = val
setval!(vi::VarInfo, val::Any, vview::Vector{UnitRange}) = length(vview) > 0 ? (vi.vals[[i for arr in vview for i in arr]] = val) : nothing

getall(vi::VarInfo)            = vi.vals
setall!(vi::VarInfo, val::Any) = vi.vals = val

getsym(vi::VarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
setgid!(vi::VarInfo, gid::Selector, vn::VarName) = push!(vi.gids[getidx(vi, vn)], gid)

istrans(vi::VarInfo, vn::VarName) = is_flagged(vi, vn, "trans")
settrans!(vi::VarInfo, trans::Bool, vn::VarName) = trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")

getlogp(vi::VarInfo) = vi.logp
setlogp!(vi::VarInfo, logp::Real) = vi.logp = logp
acclogp!(vi::VarInfo, logp::Any) = vi.logp += logp
resetlogp!(vi::VarInfo) = setlogp!(vi, zero(Real))

isempty(vi::VarInfo) = isempty(vi.idcs)

# X -> R for all variables associated with given sampler
function link!(vi::VarInfo, spl)
    vns = getvns(vi, spl)
    if ~istrans(vi, vns[1])
      for vn in vns
        dist = getdist(vi, vn)
        setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
        settrans!(vi, true, vn)
      end
    else
      @warn("[Turing] attempt to link a linked vi")
    end
end

# R -> X for all variables associated with given sampler
function invlink!(vi::VarInfo, spl::Sampler)
    vns = getvns(vi, spl)
    if istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, false, vn)
        end
    else
        @warn("[Turing] attempt to invlink an invlinked vi")
    end
end

vns(vi::VarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::VarInfo) = map(vn -> vn.sym, vns(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
function Base.getindex(vi::VarInfo, vn::VarName)
    @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vn)
    return copy(istrans(vi, vn) ?
        invlink(dist, reconstruct(dist, getval(vi, vn))) :
        reconstruct(dist, getval(vi, vn)))
end

Base.setindex!(vi::VarInfo, val::Any, vn::VarName) = setval!(vi, val, vn)

function Base.getindex(vi::VarInfo, vns::Vector{VarName})
    @assert haskey(vi, vns[1]) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vns[1])
    return copy(istrans(vi, vns[1]) ?
        invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
        reconstruct(dist, getval(vi, vns), length(vns)))
end

# NOTE: vi[vview] will just return what insdie vi (no transformations applied)
Base.getindex(vi::VarInfo, vview::VarView) = copy(getval(vi, vview))
Base.setindex!(vi::VarInfo, val::Any, vview::VarView) = setval!(vi, val, vview)

Base.getindex(vi::VarInfo, s::Union{Selector, Sampler}) = copy(getval(vi, getranges(vi, s)))
Base.setindex!(vi::VarInfo, val::Any, s::Union{Selector, Sampler}) = setval!(vi, val, getranges(vi, s))

Base.getindex(vi::VarInfo, ::SampleFromPrior) = copy(getall(vi))
Base.setindex!(vi::VarInfo, val::Any, ::SampleFromPrior) = setall!(vi, val)

Base.keys(vi::VarInfo) = keys(vi.idcs)

Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, vn)

function Base.show(io::IO, vi::VarInfo)
    vi_str = """
    /=======================================================================
    | VarInfo
    |-----------------------------------------------------------------------
    | Varnames  :   $(string(vi.vns))
    | Range     :   $(vi.ranges)
    | Vals      :   $(vi.vals)
    | RVs       :   $(vi.rvs)
    | GIDs      :   $(vi.gids)
    | Orders    :   $(vi.orders)
    | Logp      :   $(vi.logp)
    | #produce  :   $(vi.num_produce)
    | flags     :   $(vi.flags)
    \\=======================================================================
    """
    print(io, vi_str)
end

# Add a new entry to VarInfo
push!(vi::VarInfo, vn::VarName, r::Any, dist::Distributions.Distribution) = push!(vi, vn, r, dist, Set{Selector}([]))
push!(vi::VarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gid::Selector) = push!(vi, vn, r, dist, Set([gid]))
function push!(vi::VarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gidset::Set{Selector})

    @assert ~(vn in vns(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

    val = vectorize(dist, r)

    vi.idcs[vn] = length(vi.idcs) + 1
    push!(vi.vns, vn)
    l = length(vi.vals); n = length(val)
    push!(vi.ranges, l+1:l+n)
    append!(vi.vals, val)
    push!(vi.dists, dist)
    push!(vi.gids, gidset)
    push!(vi.orders, vi.num_produce)
    push!(vi.flags["del"], false)
    push!(vi.flags["trans"], false)

    return vi
end

function setorder!(vi::VarInfo, vn::VarName, index::Int)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    return vi
end

# This method is use to generate a new VarName with the right count
function VarName(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String)
    # TODO: update this method when implementing the sanity check
    VarName(csym, sym, indexing, 1)
end
function VarName(vi::VarInfo, syms::Vector{Symbol}, indexing::String)
    # TODO: update this method when implementing the sanity check
    return VarName(syms[1], syms[2], indexing, 1)
end

#################################
# Utility functions for VarInfo #
#################################

# function expand!(vi::VarInfo)
#   push!(vi.vals, vi.vals[end]); vi.vals[end], vi.vals[end-1] = vi.vals[end-1], vi.vals[end]
#   push!(vi.trans, deepcopy(vi.trans[end]))
#   push!(vi.logp, zero(Real))
# end
#
# function shrink!(vi::VarInfo)
#   pop!(vi.vals)
#   pop!(vi.trans)
#   pop!(vi.logp)
# end
#
# function last!(vi::VarInfo)
#   vi.vals = vi.vals[end:end]
#   vi.trans = vi.trans[end:end]
#   vi.logp = vi.logp[end:end]
# end

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
getidcs(vi::VarInfo, ::SampleFromPrior) = filter(i -> isempty(vi.gids[i]) , 1:length(vi.gids))
function getidcs(vi::VarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
        spl.info[:idcs]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
        spl.info[:idcs] = getidcs(vi, spl.selector, spl.alg.space)
    end
end

# Get all indices of variables belonging to a given selector
function getidcs(vi::VarInfo, s::Selector, space::Set=Set())
    filter(i -> (s in vi.gids[i] || isempty(vi.gids[i])) && (isempty(space) || is_inside(vi.vns[i], space)),
           1:length(vi.gids))
end

function is_inside(vn::VarName, space::Set)::Bool
    if vn.sym in space
        return true
    else
        exprs = filter(el -> isa(el, Expr), space)
        strs = Set((replace(string(ex), r"\(|\)" => "") for ex in exprs))
        vn_str = string(vn.sym) * vn.indexing
        valid = filter(str -> occursin(str, vn_str), strs)
        return length(valid) > 0
    end
end

# Get all values of variables belonging to spl.selector
getvals(vi::VarInfo, spl::AbstractSampler) = view(vi.vals, getidcs(vi, spl))

# Get all vns of variables belonging to spl.selector
getvns(vi::VarInfo, spl::AbstractSampler) = view(vi.vns, getidcs(vi, spl))

# Get all vns of variables belonging to spl.selector
function getranges(vi::VarInfo, spl::Sampler)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
        spl.info[:ranges]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        spl.info[:ranges] = getranges(vi, spl.selector, spl.alg.space)
    end
end

function getranges(vi::VarInfo, s::Selector, space::Set=Set())
    union(map(i -> vi.ranges[i], getidcs(vi, s, space))...)
end

# NOTE: this function below is not used anywhere but test files.
#       we can safely remove it if we want.
function getretain(vi::VarInfo, spl::AbstractSampler)
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

is_flagged(vi::VarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)]
set_flag!(vi::VarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)] = true
unset_flag!(vi::VarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)] = false

function set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)
    gidcs = getidcs(vi, spl)
    if vi.num_produce == 0
        for i = length(gidcs):-1:1
          vi.flags["del"][gidcs[i]] = true
        end
    else
        retained = [idx for idx in 1:length(vi.orders) if idx in gidcs && vi.orders[idx] > vi.num_produce]
        for i = retained
            vi.flags["del"][i] = true
        end
    end
end

function updategid!(vi::VarInfo, vn::VarName, spl::Sampler)
    if ~isempty(spl.alg.space) && isempty(getgid(vi, vn)) && getsym(vi, vn) in spl.alg.space
        setgid!(vi, spl.selector, vn)
    end
end

end
