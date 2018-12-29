const Vec{T} = AbstractVector{T}

struct SingleVarInfo{sym, T, TDist <: Distribution, TDists <: Vec{TDist}, TVN <: Vec{VarName{sym}}, TVal <: Vec{T}, TRanges <: Vec{UnitRange{Int}}, TId <: Vec{Int}}
    idcs        ::    Dict{VarName{sym}, Int}
    vns         ::    TVN
    ranges      ::    TRanges
    vals        ::    TVal
    dists       ::    TDists
    gids        ::    TId
    orders      ::    TId   # observe statements number associated with random variables
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
        Vector{Int}(),
        Vector{Int}(),
        flags
    )
end
getdisttype(::SingleVarInfo{<:Any, <:Any, TDist}) where TDist = TDist

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
        _vals = [vi.vals[_ranges[i]] for i in sym_inds]
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

function getidx(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).idcs[vn]
end
function getrange(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).ranges[getidx(vi, vn)]
end
function getranges(vi::TypedVarInfo, vns::Vector{VarName{sym}}) where sym
    union(map(vn -> getrange(vi, vn), vns)...)
end
function getval(vi::TypedVarInfo, vn::VarName{sym}) where sym
    view(getfield(vi.vis, sym).vals, getrange(vi, vn))
end
function setval!(vi::TypedVarInfo, val, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).vals[getrange(vi, vn)] = val
end
function getval(vi::TypedVarInfo, vns::Vector{VarName{sym}}) where sym
    view(getfield(vi.vis, sym).vals, getranges(vi, vns))
end
@generated function getall(vi::TypedVarInfo{Tvis}) where Tvis
    vals = [:(vi.vis.$f.vals) for f in fieldnames(Tvis)]
    return Expr(:call, :append, vals...)
end
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
function getsym(vi::TypedVarInfo, vn::VarName{sym}) where sym
    if isdefined(vi.vis, sym)
        return sym
    else
        error("$sym not defined in the TypedVarInfo instance.")
    end
end
function getdist(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).dists[getidx(vi, vn)]
end
function getgid(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).gids[getidx(vi, vn)]
end

function setgid!(vi::TypedVarInfo, gid::Int, vn::VarName{sym}) where sym
    getfield(vi.vis, sym).gids[getidx(vi, vn)] = gid
end
@generated function isempty(vi::TypedVarInfo{Tvis}) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :(isempty(vi.vis.$f.idcs)))
    end
    return Expr(:&&, args...)
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
syms(vi::TypedVarInfo) = fieldnames(vi.vis)

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

NewVarInfo(old_vi::UntypedVarInfo, spl, x) = old_vi
NewVarInfo(old_vi::TypedVarInfo, spl, x) = newvarinfo(old_vi, spl, x)
@generated function newvarinfo(old_vi::TypedVarInfo{Tvis}, spl::S, x::Type{T}) where {Tvis, T, S}
    syms = getspace(S)
    vi = :(old_vi.vis)
    args = []
    for f in fieldnames(Tvis)
        if f ∈ syms || length(syms) == 0
            idcs, vns = :($vi.$f.idcs), :($vi.$f.vns)
            ranges = :($vi.$f.ranges)
            vals = quote
                if eltype($vi.$f.vals) === $T
                    $vi.$f.vals
                else
                    new_vals = similar($vi.$f.vals, $T)
                    new_vals .= $vi.$f.vals
                end
            end
            ranges, vals = :($vi.$f.ranges), :(similar($vi.$f.vals, $T))
            dists, gids = :($vi.$f.dists), :($vi.$f.gids)
            orders, flags = :($vi.$f.orders), :($vi.$f.flags)
            arg = :($f = SingleVarInfo($idcs, $vns, $ranges, $vals, $dists, $gids, $orders, $flags))
            push!(args, arg)
        else
            push!(args, :($f = $vi.$f))
        end
    end
    if length(args) == 0
        return :(TypedVarInfo(NamedTuple(), Ref(old_vi.logp*one($T)), Ref(old_vi.num_produce)))
    else
        return :(TypedVarInfo(($(args...),), Ref(old_vi.logp*one($T)), Ref(old_vi.num_produce)))
    end
end

function Base.haskey(vi::TypedVarInfo{Tvis}, vn::VarName{sym}) where {Tvis, sym}
    return sym in fieldnames(Tvis) && haskey(getfield(vi.vis, sym).idcs, vn)    
end
function push!(
            mvi::TypedVarInfo, 
            vn::VarName{sym}, 
            r::Any, 
            dist::Distributions.Distribution, 
            gid::Int
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
    push!(vi.gids, gid)
    push!(vi.orders, mvi.num_produce)
    push!(vi.flags["del"], false)
    push!(vi.flags["trans"], false)

    return vi
end

function setorder!(mvi::TypedVarInfo, vn::VarName{sym}, index::Int) where {sym}
    vi = getfield(mvi.vis, sym)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    mvi
end

_filter_gids_1(vi, f) = filter(i -> getfield(vi.vis, f).gids[i] == 0, 1:length(getfield(vi.vis, f).gids))
@generated function getidcs(vi::TypedVarInfo{Tvis}, spl::Nothing) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _filter_gids_1(vi, $(QuoteNode(f)))))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end
    return nt
end

function _filter_gids_2(mvi, spl, f)
    vi = getfield(mvi.vis, f)
    return filter(i -> (vi.gids[i] == spl.alg.gid || vi.gids[i] == 0) && (isempty(getspace(spl)) || is_inside(vi.vns[i], getspace(spl))), 1:length(vi.gids))
end
@generated function getidcs(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _filter_gids_2(vi, spl, $(QuoteNode(f)))))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end

    return quote
        # NOTE: 0b00 is the sanity flag for
        #         |\____ getidcs   (mask = 0b10)
        #         \_____ getranges (mask = 0b01)
        if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
        if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
            spl.info[:idcs]
        else
            spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
            spl.info[:idcs] = $nt
        end
    end
end

@generated function getvns(vi::TypedVarInfo{Tvis}, spl::Union{Nothing, Sampler}) where Tvis
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

_map(vi, spl, f, idcs) = union(map(i -> getfield(vi.vis, f).ranges[i], idcs)..., Int[])
@generated function getranges(vi::TypedVarInfo{Tvis}, spl::Sampler) where Tvis
    args = []
    for f in fieldnames(Tvis)
        push!(args, :($f = _map(vi, spl, $(QuoteNode(f)), idcs.$f)))
    end
    if length(args) == 0
        nt = :(NamedTuple())
    else
        nt = :(($(args...),))
    end

    return quote
        idcs = getidcs(vi, spl)
        if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
        if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
            spl.info[:ranges]
        else
            spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
            spl.info[:ranges] = $nt
        end
    end
end
function is_flagged(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)]
end
function set_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)] = true
end
function unset_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    getfield(vi.vis, sym).flags[flag][getidx(vi, vn)] = false
end

function get_retained(orders, gidcs, num_produce)
    [idx for idx in 1:length(orders) if idx in gidcs && orders[idx] > num_produce]
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
