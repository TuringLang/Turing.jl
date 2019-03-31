module RandomVariables

using ...Turing: Turing, CACHERESET, CACHEIDCS, CACHERANGES, Model,
    AbstractSampler, Sampler, SampleFromPrior,
    Selector
using ...Utilities: vectorize, reconstruct, reconstruct!
using Bijectors: SimplexDistribution
using Distributions

import Base: string, isequal, ==, hash, getindex, setindex!, push!, show, isempty
import Turing: link, invlink

export  VarName,
        AbstractVarInfo,
        VarInfo,
        UntypedVarInfo,
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

sym_idx(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol
getsym(vn::VarName{sym}) where sym = sym

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

copybyindex(vn::VarName, indexing::String) = VarName(vn.csym, vn.sym, indexing, vn.counter)

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

###################
# Untyped VarInfo #
###################

abstract type AbstractVarInfo end
const VarInfo = AbstractVarInfo

function Turing.runmodel!(model::Model, vi::AbstractVarInfo, spl::AbstractSampler = SampleFromPrior())
    setlogp!(vi, zero(Float64))
    if spl isa Sampler && haskey(spl.info, :eval_num)
        spl.info[:eval_num] += 1
    end
    model(vi, spl)
    return vi
end

mutable struct UntypedVarInfo <: AbstractVarInfo
    idcs        ::    Dict{VarName,Int}
    vns         ::    Vector{VarName}
    ranges      ::    Vector{UnitRange{Int}}
    vals        ::    Vector{Real}
    dists       ::    Vector{Distributions.Distribution}
    gids        ::    Vector{Set{Selector}}
    logp        ::    Real
    num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
    orders      ::    Vector{Int}   # observe statements number associated with random variables
    flags       ::    Dict{String, BitVector}

    UntypedVarInfo() = begin
        vals  = Vector{Real}()
        logp  = 0.0
        flags = Dict{String, BitVector}()
        flags["del"] = BitVector()
        flags["trans"] = BitVector()

        new(
            Dict{VarName, Int}(),
            Vector{VarName}(),
            Vector{UnitRange{Int}}(),
            vals,
            Vector{Distributions.Distribution}(),
            Vector{Set{Selector}}(),
            logp,
            0,
            Vector{Int}(),
            flags
        )
    end
end
VarInfo() = UntypedVarInfo()

###########################
# Single variable VarInfo #
###########################

struct SingleVarInfo{sym, T, TDist <: Distribution, TDists <: AbstractVector{TDist}, TVN <: AbstractVector{VarName{sym}}, TVal <: AbstractVector{T}, TRanges <: AbstractVector{UnitRange{Int}}, TId <: AbstractVector{Set{Selector}}, Torders <: AbstractVector{Int}}
    idcs        ::    Dict{VarName{sym}, Int}
    vns         ::    TVN
    ranges      ::    TRanges
    vals        ::    TVal
    dists       ::    TDists
    gids        ::    TId
    orders      ::    Torders   # observe statements number associated with random variables
    flags       ::    Dict{String, BitVector}
end
SingleVarInfo{sym, T}() where {sym, T} = SingleVarInfo{sym, T, Distribution}()
function SingleVarInfo{sym, T, TDist}() where {sym, T, TDist}
    vals  = Vector{T}()
    flags = Dict{String, BitVector}()
    flags["del"] = BitVector()
    flags["trans"] = BitVector()

    SingleVarInfo(
        Dict{VarName{sym}, Int}(),
        Vector{VarName{sym}}(),
        Vector{UnitRange{Int}}(),
        vals,
        Vector{TDist}(),
        Vector{Set{Selector}}(),
        Vector{Int}(),
        flags
    )
end

function Base.empty!(vi::SingleVarInfo)
    empty!(vi.idcs)
    empty!(vi.vns)
    empty!(vi.ranges)
    empty!(vi.vals)
    empty!(vi.dists)
    empty!(vi.gids)
    empty!(vi.orders)
    for k in keys(vi.flags)
        empty!(vi.flags[k])
    end

    return vi
end

#######################
# Fully typed VarInfo #
#######################

struct TypedVarInfo{Tvis, Tlogp} <: AbstractVarInfo
    vis::Tvis
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
@generated function TypedVarInfo(vis::Tvis) where {Tvis <: Tuple}
    syms = []
    Ts = []
    args = []
    for (i, T) in enumerate(Tvis.types)
        push!(Ts, T)
        sym = T.parameters[1]
        push!(syms, sym)
        push!(args, :($sym = vis[$i]))
    end
    nt = length(args) == 0 ? :(NamedTuple()) : :(($(args...),))

    return :(TypedVarInfo{NamedTuple{$(Tuple(syms)), $(Tuple{Ts...})}, Float64}($nt, Ref(0.0), Ref(0)))
end
function TypedVarInfo(vi::UntypedVarInfo)
    vis = SingleVarInfo[]
    syms_tuple = Tuple(syms(vi))
    for s in syms_tuple
        _inds = findall(vn -> vn.sym == s, vi.vns)
        sym_inds = collect(1:length(_inds))
        sym_vns = getindex.((vi.vns,), _inds)
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        sym_dists = getindex.((vi.dists,), _inds)
        sym_gids = getindex.((vi.gids,), _inds)
        sym_orders = getindex.((vi.orders,), _inds)
        sym_flags = Dict(a => vi.flags[a][_inds] for a in keys(vi.flags))

        _ranges = getindex.((vi.ranges,), _inds)
        # `copy` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(vi.vals[_ranges[i]]) for i in sym_inds]
        sym_ranges = Vector{eltype(_ranges)}(undef, length(sym_inds))
        start = 0
        for i in sym_inds
            sym_ranges[i] = start + 1 : start + length(_vals[i])
            start += length(_vals[i])
        end
        sym_vals = foldl(vcat, _vals)

        push!(vis, 
            SingleVarInfo(
                            sym_idcs,
                            sym_vns,
                            sym_ranges,
                            sym_vals,
                            sym_dists,
                            sym_gids,
                            sym_orders,
                            sym_flags,        
            )
        )
    end
    logp = vi.logp
    num_produce = vi.num_produce
    vis_tuple = Tuple(vis)
    vis_nt = NamedTuple{syms_tuple, Tuple{typeof.(vis_tuple)...}}(vis_tuple)
    return TypedVarInfo(vis_nt, Ref(logp), Ref(num_produce))
end

function Base.getproperty(vi::TypedVarInfo, f::Symbol)
    f === :logp && return getfield(vi, :logp)[]
    f === :num_produce && return getfield(vi, :num_produce)[]
    return getfield(vi, f)
end
function Base.setproperty!(vi::TypedVarInfo, f::Symbol, x)
    f === :logp && return getfield(vi, :logp)[] = x
    f === :num_produce && return getfield(vi, :num_produce)[] = x
    return setfield!(vi, f, x)
end

@generated function Base.empty!(vi::TypedVarInfo{Tvis}) where Tvis
    expr = Expr(:block)
    for f in fieldnames(Tvis)
        push!(expr.args, :(empty!(vi.vis.$f)))
    end
    push!(expr.args, quote
        vi.logp = 0
        vi.num_produce = 0
        return vi
    end)
    return expr
end

#####################
# Utility functions #
#####################

# Functions defined only for UntypedVarInfo
vns(vi::UntypedVarInfo) = Set(keys(vi.idcs)) # get all vns
Base.keys(vi::UntypedVarInfo) = keys(vi.idcs)
const VarView = Union{Int,UnitRange,Vector{Int},Vector{UnitRange}}
getval(vi::UntypedVarInfo, vview::VarView) = view(vi.vals, vview)
setval!(vi::UntypedVarInfo, val, vview::VarView) = vi.vals[vview] = val
function setval!(vi::UntypedVarInfo, val, vview::Vector{UnitRange})
    if length(vview) > 0
        return (vi.vals[[i for arr in vview for i in arr]] = val)
    else
        return nothing
    end
end

getidx(vi::UntypedVarInfo, vn::VarName) = vi.idcs[vn]
function getidx(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).idcs[vn]
end

getrange(vi::UntypedVarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]
function getrange(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).ranges[getidx(vi, vn)]
end
function getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return union(map(vn -> getrange(vi, vn), vns)...)
end

getval(vi::UntypedVarInfo, vn::VarName) = view(vi.vals, getrange(vi, vn))
function getval(vi::TypedVarInfo, vn::VarName{sym}) where sym
    view(getfield(vi.vis, sym).vals, getrange(vi, vn))
end
setval!(vi::UntypedVarInfo, val, vn::VarName) = vi.vals[getrange(vi, vn)] = val
function setval!(vi::TypedVarInfo, val, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).vals[getrange(vi, vn)] = val
end

getval(vi::UntypedVarInfo, vns::Vector{<:VarName}) = view(vi.vals, getranges(vi, vns))
function getval(vi::TypedVarInfo, vns::Vector{VarName{sym}}) where sym
    view(getfield(vi.vis, sym).vals, getranges(vi, vns))
end

getall(vi::UntypedVarInfo) = vi.vals
@generated function getall(vi::TypedVarInfo{Tvis}) where Tvis
    vals = [:(vi.vis.$f.vals) for f in fieldnames(Tvis)]
    return Expr(:call, :append, vals...)
end
setall!(vi::UntypedVarInfo, val) = vi.vals = val
@generated function setall!(vi::TypedVarInfo{Tvis}, val) where Tvis
    expr = Expr(:block)
    ranges = Dict{Symbol, Expr}
    start = 0
    for f in fieldnames(Tvis)
        push!(ranges, :($start + 1 : $start + length(vi.vis.$f.vals)))
        start = :($start + length(vi.vis.$f.vals))
    end
    return [:(vi.vis.$f.vals .= @view val[$(ranges[f])]) for f in fieldnames(Tvis)]
end

getsym(vi::UntypedVarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym
function getsym(vi::TypedVarInfo, vn::VarName{sym}) where sym
    if isdefined(vi.vis, sym)
        return sym
    else
        error("$sym not defined in the TypedVarInfo instance.")
    end
end

getdist(vi::UntypedVarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
function getdist(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).dists[getidx(vi, vn)]
end

getgid(vi::UntypedVarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
function getgid(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).gids[getidx(vi, vn)]
end

setgid!(vi::UntypedVarInfo, gid::Selector, vn::VarName) = push!(vi.gids[getidx(vi, vn)], gid)
function setgid!(vi::TypedVarInfo, gid::Selector, vn::VarName{sym}) where sym
    push!(getfield(vi.vis, sym).gids[getidx(vi, vn)], gid)
end

istrans(vi::AbstractVarInfo, vn::VarName) = is_flagged(vi, vn, "trans")
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end

getlogp(vi::AbstractVarInfo) = vi.logp
setlogp!(vi::AbstractVarInfo, logp::Real) = vi.logp = logp
acclogp!(vi::AbstractVarInfo, logp::Any) = vi.logp += logp
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, 0.0)

isempty(vi::UntypedVarInfo) = isempty(vi.idcs)
@generated function isempty(vi::TypedVarInfo{Tvis}) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :(isempty(vi.vis.$f.idcs)))
    end
    return Expr(:&&, args...)
end

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
@generated function link!(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    expr = Expr(:block, :(vns = getvns(vi, spl)))
    space = getspace(spl)
    for f in fieldnames(Tvis)
        if f ∈ space || length(space) == 0
            push!(expr.args, quote
                if ~istrans(vi, vns.$f[1])
                    for vn in vns.$f
                        dist = getdist(vi, vn)
                        setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
                        settrans!(vi, true, vn)
                    end
                else
                    @warn("[Turing] attempt to link a linked vi")
                end
            end)
        end
    end
    return expr
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
@generated function invlink!(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    expr = Expr(:block, :(vns = getvns(vi, spl)))
    space = getspace(spl)
    for f in fieldnames(Tvis)
        if f ∈ space || length(space) == 0
            push!(expr.args, quote
                if istrans(vi, vns.$f[1])
                    for vn in vns.$f
                        dist = getdist(vi, vn)
                        setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
                        settrans!(vi, false, vn)
                    end
                else
                    @warn("[Turing] attempt to invlink an invlinked vi")
                end
            end)
        end
    end
    return expr
end

syms(vi::UntypedVarInfo) = unique!(map(vn -> vn.sym, vi.vns))  # get all symbols
syms(vi::TypedVarInfo) = fieldnames(vi.vis)

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

Base.getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, getranges(vi, spl)))
@generated function Base.getindex(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :(vi.vis.$f.vals[ranges.$f]))
    end

    return quote
        ranges = getranges(vi, spl)
        return vcat($(args...))
    end
end

Base.setindex!(vi::UntypedVarInfo, val::Any, spl::Sampler) = setval!(vi, val, getranges(vi, spl))
@generated function Base.setindex!(vi::TypedVarInfo{Tvis}, val::Any, spl::Sampler) where Tvis
    expr = Expr(:block)
    push!(expr.args, :(start = 0))
    for f in fieldnames(Tvis)
        push!(expr.args, quote
            r = @views vi.vis.$f.vals[ranges.$f]
            v = @views val[start + 1 : start + length(r)]
            n = length(v)
            vi.vis.$f.vals[ranges.$f] .= v
            start += length(r)
        end)
    end

    return quote
        ranges = getranges(vi, spl)
        $expr
        return val
    end
end

Base.getindex(vi::AbstractVarInfo, spl::SampleFromPrior) = copy(getall(vi))
Base.setindex!(vi::AbstractVarInfo, val::Any, spl::SampleFromPrior) = setall!(vi, val)

Base.haskey(vi::UntypedVarInfo, vn::VarName) = haskey(vi.idcs, vn)
function Base.haskey(vi::TypedVarInfo{Tvis}, vn::VarName{sym}) where {Tvis, sym}
    return sym in fieldnames(Tvis) && haskey(getfield(vi.vis, sym).idcs, vn)    
end

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
push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution) = push!(vi, vn, r, dist, Set{Selector}([]))
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, spl::Sampler)
    spl.info[:cache_updated] = CACHERESET
    push!(vi, vn, r, dist, spl.selector)
end
push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, spl::AbstractSampler) = push!(vi, vn, r, dist)
push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gid::Selector) = push!(vi, vn, r, dist, Set([gid]))
function push!(vi::UntypedVarInfo, vn::VarName, r::Any, dist::Distributions.Distribution, gidset::Set{Selector})
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
function push!(
            mvi::TypedVarInfo, 
            vn::VarName{sym}, 
            r::Any, 
            dist::Distributions.Distribution, 
            gidset::Set{Selector}
            ) where sym
    
    @assert ~(haskey(mvi, vn)) "[push!] attempt to add an exisitng variable $(vn.sym) ($(vn)) to TypedVarInfo of syms $(syms(mvi)) with dist=$dist, gid=$gid"

    val = vectorize(dist, r)

    vi = getfield(mvi.vis, sym)
    vi.idcs[vn] = length(vi.idcs) + 1
    push!(vi.vns, vn)
    l = length(vi.vals); n = length(val)
    push!(vi.ranges, l+1:l+n)
    append!(vi.vals, val)
    push!(vi.dists, dist)
    push!(vi.gids, gidset)
    push!(vi.orders, mvi.num_produce)
    push!(vi.flags["del"], false)
    push!(vi.flags["trans"], false)

    return vi
end

function setorder!(vi::UntypedVarInfo, vn::VarName, index::Int)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    return vi
end
function setorder!(mvi::TypedVarInfo, vn::VarName{sym}, index::Int) where {sym}
    vi = getfield(mvi.vis, sym)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    return mvi
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
getidcs(vi::UntypedVarInfo, ::SampleFromPrior) = filter(i -> isempty(vi.gids[i]) , 1:length(vi.gids))
@generated function getidcs(vi::TypedVarInfo{Tvis}, spl::SampleFromPrior) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _filter_gids(vi, $(QuoteNode(f)))))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end
    return nt
end
function _filter_gids(vi, f::Symbol)
    filter(i -> getfield(vi.vis, f).gids[i] == 0, 1:length(getfield(vi.vis, f).gids))
end
function getidcs(vi::AbstractVarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         | \___ getidcs   (mask = 0b10)
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
function getidcs(vi::UntypedVarInfo, s::Selector, space)
    filter(i -> (s in vi.gids[i] || isempty(vi.gids[i])) && (isempty(space) || is_inside(vi.vns[i], space)),
           1:length(vi.gids))
end
@generated function getidcs(vi::TypedVarInfo{Tvis}, s::Selector, space) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _filter_gids(vi, s, $(QuoteNode(f)), space)))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end
    return nt
end
function _filter_gids(mvi, s::Selector, f::Symbol, space)
    vi = getfield(mvi.vis, f)
    function func(i)
        return (s in vi.gids[i] || isempty(vi.gids[i])) && 
            (isempty(space) || is_inside(vi.vns[i], space))
    end
    return filter(func, 1:length(vi.gids))
end

# Get all vns of variables belonging to spl.selector
getvns(vi::UntypedVarInfo, spl::AbstractSampler) = view(vi.vns, getidcs(vi, spl))
@generated function getvns(vi::TypedVarInfo{Tvis}, spl::AbstractSampler) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = vi.vis.$f.vns[idcs.$f]))
    end
    nt = length(args) == 0 ? :(NamedTuple()) : :(($(args...),))

    return quote
        idcs = getidcs(vi, spl)
        return $nt
    end
end

# Get all vns of variables belonging to spl.selector
function getranges(vi::AbstractVarInfo, spl::Sampler)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
        spl.info[:ranges]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        spl.info[:ranges] = getranges(vi, spl.selector, spl.alg.space)
    end
end
function getranges(vi::AbstractVarInfo, s::Selector, space::Set=Set())
    _getranges(vi, getidcs(vi, s, space))
end
function _getranges(vi::UntypedVarInfo, idcs)
    union(map(i -> vi.ranges[i], idcs)...)
end
@generated function _getranges(vi::TypedVarInfo{Tvis}, idcs) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _map(vi, $(QuoteNode(f)), idcs.$f)))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end
    return nt
end
_map(vi, f, idcs) = union(map(i -> getfield(vi.vis, f).ranges[i], idcs)..., Int[])

#######################################
# Rand & replaying method for VarInfo #
#######################################

function is_flagged(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)]
end
function is_flagged(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)]
end

function set_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)] = true
end
function set_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)] = true
end

function unset_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)] = false
end
function unset_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)] = false
end

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
@generated function set_retained_vns_del_by_spl!(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    branch1_expr = Expr(:block)
    branch2_expr = Expr(:block)
    for f in fieldnames(Tvis)
        push!(branch1_expr.args, quote
            for i = length(gidcs.$f):-1:1
                vi.vis.$f.flags["del"][gidcs.$f[i]] = true
            end
        end)
        push!(branch2_expr.args, quote
            retained = get_retained(vi.vis.$f.orders, gidcs.$f, vi.num_produce)
            for i in retained
                vi.vis.$f.flags["del"][i] = true
            end
        end)
    end

    return quote
        gidcs = getidcs(vi, spl)
        if vi.num_produce == 0
            $branch1_expr
        else
            $branch2_expr
        end
    end
end
function get_retained(orders, gidcs, num_produce)
    [idx for idx in 1:length(orders) if idx in gidcs && orders[idx] > num_produce]
end

function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler)
    if ~isempty(spl.alg.space) && isempty(getgid(vi, vn)) && getsym(vi, vn) in spl.alg.space
        setgid!(vi, spl.selector, vn)
    end
end

end
