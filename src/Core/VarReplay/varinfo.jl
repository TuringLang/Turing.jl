###########
# VarName #
###########
struct VarName{sym}
    csym      ::    Symbol        # symbol generated in compilation time
    indexing  ::    String        # indexing
    counter   ::    Int           # counter of same {csym, uid}
end
VarName(csym, sym, indexing, counter) = VarName{sym}(csym, indexing, counter)

function Base.getproperty(vn::VarName{sym}, f::Symbol) where {sym}
    return f === :sym ? sym : getfield(vn, f)
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)
Base.hash(vn::VarName) = hash(uid(vn))

isequal(x::VarName, y::VarName) = hash(uid(x)) == hash(uid(y))
==(x::VarName, y::VarName)      = isequal(x, y)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{<:VarName}) = replace(string(map(vn -> string(vn), vns)), "String" => "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

copybyindex(vn::VarName, indexing::String) = VarName(vn.csym, vn.sym, indexing, vn.counter)

###########
# VarInfo #
###########

abstract type AbstractVarInfo end
const VarInfo = AbstractVarInfo

mutable struct UntypedVarInfo <: AbstractVarInfo
    idcs        ::    Dict{VarName,Int}
    vns         ::    Vector{VarName}
    ranges      ::    Vector{UnitRange{Int}}
    vals        ::    Vector{Real}
    dists       ::    Vector{Distributions.Distribution}
    gids        ::    Vector{Int}
    logp        ::    Real
    num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
    orders      ::    Vector{Int}   # observe statements number associated with random variables
    flags       ::    Dict{String, BitVector}

    UntypedVarInfo() = begin
        vals  = Vector{Real}()
        logp  = zero(Real)
        flags = Dict{String, BitVector}()
        flags["del"] = BitVector()
        flags["trans"] = BitVector()

        new(
            Dict{VarName, Int}(),
            Vector{VarName}(),
            Vector{UnitRange{Int}}(),
            vals,
            Vector{Distributions.Distribution}(),
            Vector{Int}(),
            logp,
            0,
            Vector{Int}(),
            flags
        )
    end
end
VarInfo() = UntypedVarInfo()

const VarView = Union{Int,UnitRange,Vector{Int},Vector{UnitRange}}

getidx(vi::UntypedVarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::UntypedVarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
getranges(vi::UntypedVarInfo, vns::Vector{<:VarName}) = union(map(vn -> getrange(vi, vn), vns)...)

getval(vi::UntypedVarInfo, vn::VarName)       = view(vi.vals, getrange(vi, vn))
setval!(vi::UntypedVarInfo, val, vn::VarName) = vi.vals[getrange(vi, vn)] = val

getval(vi::UntypedVarInfo, vns::Vector{<:VarName}) = view(vi.vals, getranges(vi, vns))

getval(vi::UntypedVarInfo, vview::VarView)                      = view(vi.vals, vview)
setval!(vi::UntypedVarInfo, val::Any, vview::VarView)           = vi.vals[vview] = val
setval!(vi::UntypedVarInfo, val::Any, vview::Vector{UnitRange}) = length(vview) > 0 ? (vi.vals[[i for arr in vview for i in arr]] = val) : nothing

getall(vi::UntypedVarInfo)            = vi.vals
setall!(vi::UntypedVarInfo, val::Any) = vi.vals = val

getsym(vi::UntypedVarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym

getdist(vi::UntypedVarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]

getgid(vi::UntypedVarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]

setgid!(vi::UntypedVarInfo, gid::Int, vn::VarName) = vi.gids[getidx(vi, vn)] = gid

istrans(vi::AbstractVarInfo, vn::VarName) = is_flagged(vi, vn, "trans")
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end

getlogp(vi::AbstractVarInfo) = vi.logp
setlogp!(vi::AbstractVarInfo, logp::Real) = vi.logp = logp
acclogp!(vi::AbstractVarInfo, logp::Any) = vi.logp += logp
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, zero(Real))

isempty(vi::UntypedVarInfo) = isempty(vi.idcs)

# X -> R for all variables associated with given sampler
function link!(vi::UntypedVarInfo, spl::Sampler)
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
function invlink!(vi::UntypedVarInfo, spl::Sampler)
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

vns(vi::UntypedVarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::UntypedVarInfo) = map(vn -> vn.sym, vi.vns)  # get all symbols

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
function Base.getindex(vi::AbstractVarInfo, vn::VarName)
    @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vn)
    return copy(istrans(vi, vn) ?
        invlink(dist, reconstruct(dist, getval(vi, vn))) :
        reconstruct(dist, getval(vi, vn)))
end

Base.setindex!(vi::AbstractVarInfo, val::Any, vn::VarName) = setval!(vi, val, vn)

function Base.getindex(vi::AbstractVarInfo, vns::Vector{<:VarName})
    @assert haskey(vi, vns[1]) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vns[1])
    return copy(istrans(vi, vns[1]) ?
        invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
        reconstruct(dist, getval(vi, vns), length(vns)))
end

# NOTE: vi[vview] will just return what insdie vi (no transformations applied)
Base.getindex(vi::UntypedVarInfo, vview::VarView) = copy(getval(vi, vview))
Base.setindex!(vi::UntypedVarInfo, val::Any, vview::VarView) = setval!(vi, val, vview)

Base.getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, getranges(vi, spl)))
Base.setindex!(vi::UntypedVarInfo, val::Any, spl::Sampler) = setval!(vi, val, getranges(vi, spl))

Base.getindex(vi::UntypedVarInfo, spl::Nothing) = copy(getall(vi))
Base.setindex!(vi::UntypedVarInfo, val::Any, spl::Nothing) = setall!(vi, val)

Base.keys(vi::UntypedVarInfo) = keys(vi.idcs)

Base.haskey(vi::UntypedVarInfo, vn::VarName) = haskey(vi.idcs, vn)

function Base.show(io::IO, vi::UntypedVarInfo)
    vi_str = """
    /=======================================================================
    | VarInfo
    |-----------------------------------------------------------------------
    | Varnames  :   $(string(vi.vns))
    | Range     :   $(vi.ranges)
    | Vals      :   $(vi.vals)
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
function push!(vi::UntypedVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gid::Int)
    @assert ~(vn in vns(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

    val = vectorize(dist, r)

    vi.idcs[vn] = length(vi.idcs) + 1
    push!(vi.vns, vn)
    l = length(vi.vals); n = length(val)
    push!(vi.ranges, l+1:l+n)
    append!(vi.vals, val)
    push!(vi.dists, dist)
    push!(vi.gids, gid)
    push!(vi.orders, vi.num_produce)
    push!(vi.flags["del"], false)
    push!(vi.flags["trans"], false)

    return vi
end

function setorder!(vi::UntypedVarInfo, vn::VarName, index::Int)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    vi
end

# This method is use to generate a new VarName with the right count
function VarName(vi::AbstractVarInfo, csym::Symbol, sym::Symbol, indexing::String)
    # TODO: update this method when implementing the sanity check
    VarName{sym}(csym, indexing, 1)
end
function VarName(vi::AbstractVarInfo, syms::Vector{Symbol}, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
      VarName{syms[2]}(syms[1], indexing, 1)
end
function VarName{sym}(vi::AbstractVarInfo, csym::Symbol, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
    VarName{sym}(csym, indexing, 1)
end

#################################
# Utility functions for VarInfo #
#################################

# Get all indices of variables belonging to gid or 0
getidcs(vi::AbstractVarInfo) = getidcs(vi, nothing)
getidcs(vi::UntypedVarInfo, spl::Nothing) = filter(i -> vi.gids[i] == 0, 1:length(vi.gids))
function getidcs(vi::UntypedVarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
        spl.info[:idcs]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
        spl.info[:idcs] = filter(i ->
            (vi.gids[i] == spl.alg.gid || vi.gids[i] == 0) && (isempty(getspace(spl)) || is_inside(vi.vns[i], getspace(spl))),
            collect(1:length(vi.gids))
        )
    end
end

@inline function _filter(f, space)
    length(space) == 0 && return ()
    if f(space[1])
        return (space[1], _filter(f, Base.tail(space))...)
    else
        return _filter(f, Base.tail(space))
    end
end
function is_inside(vn::VarName, space)
    if vn.sym in space
        return true
    else
        exprs = _filter(el -> isa(el, Expr), space)
        strs = ntuple(i -> replace(string(exprs[i]), r"\(|\)" => ""), Val(length(exprs)))
        vn_str = string(vn.sym) * vn.indexing
        valid = _filter(str -> occursin(str, vn_str), strs)
        return length(valid) > 0
    end
end

# Get all values of variables belonging to gid or 0
getvals(vi::UntypedVarInfo) = getvals(vi, nothing)
getvals(vi::UntypedVarInfo, spl::Union{Nothing, Sampler}) = view(vi.vals, getidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
getvns(vi::UntypedVarInfo) = getvns(vi, nothing)
getvns(vi::UntypedVarInfo, spl::Union{Nothing, Sampler}) = view(vi.vns, getidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
function getranges(vi::UntypedVarInfo, spl::Sampler)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
        spl.info[:ranges]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        spl.info[:ranges] = union(map(i -> vi.ranges[i], getidcs(vi, spl))...)
    end
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

is_flagged(vi::UntypedVarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)]
set_flag!(vi::UntypedVarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)] = true
unset_flag!(vi::UntypedVarInfo, vn::VarName, flag::String) = vi.flags[flag][getidx(vi, vn)] = false

function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
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

function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler)
    if ~isempty(getspace(spl)) && getgid(vi, vn) == 0 && getsym(vi, vn) in getspace(spl)
        setgid!(vi, spl.alg.gid, vn)
    end
end
