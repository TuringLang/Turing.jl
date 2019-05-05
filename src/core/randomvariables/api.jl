"""
`VarName(csym, sym, indexing, counter)`
`VarName{sym}(csym::Symbol, indexing::String)`

Constructs a new instance of `VarName{sym}`
"""
VarName(csym, sym, indexing, counter) = VarName{sym}(csym, indexing, counter)
function VarName(csym::Symbol, sym::Symbol, indexing::String)
    # TODO: update this method when implementing the sanity check
    return VarName{sym}(csym, indexing, 1)
end
function VarName{sym}(csym::Symbol, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
    return VarName{sym}(csym, indexing, 1)
end

"""
`VarName(syms::Vector{Symbol}, indexing::String)`

Constructs a new instance of `VarName{syms[2]}`
"""
function VarName(syms::Vector{Symbol}, indexing::String) where {sym}
    # TODO: update this method when implementing the sanity check
    return VarName{syms[2]}(syms[1], indexing, 1)
end

"""
`VarName(vn::VarName, indexing::String)`

Returns a copy of `vn` with a new index `indexing`.
"""
function VarName(vn::VarName, indexing::String)
    return VarName(vn.csym, vn.sym, indexing, vn.counter)
end

function getproperty(vn::VarName{sym}, f::Symbol) where {sym}
    return f === :sym ? sym : getfield(vn, f)
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

"""
`uid(vn::VarName)`

Returns a unique tuple identifier for `vn`.
"""
uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)

hash(vn::VarName) = hash(uid(vn))

==(x::VarName, y::VarName) = hash(uid(x)) == hash(uid(y))

function string(vn::VarName; all = true)
    if all
        return "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
    else
        return "$(vn.sym)$(vn.indexing)"
    end
end
function string(vns::Vector{<:VarName})
    return replace(string(map(vn -> string(vn), vns)), "String" => "")
end

"""
`Symbol(vn::VarName)`

Returns a `Symbol` represenation of the variable identifier `VarName`.
"""
Symbol(vn::VarName) = Symbol(string(vn, all=false))  # simplified symbol

"""
`in(vn::VarName, space::Set)`

Returns `true` if `vn`'s symbol is in `space` and `false` otherwise.
"""
function in(vn::VarName, space::Set)::Bool
    if vn.sym in space
        return true
    else
        # Collect expressions from space
        exprs = filter(el -> isa(el, Expr), space)
        # Filter `(` and `)` out and get a string representation of `exprs`
        expr_strs = Set((replace(string(ex), r"\(|\)" => "") for ex in exprs))
        # String representation of `vn`
        vn_str = string(vn, all=false)
        # Check if `vn_str` is in `expr_strs`
        valid = filter(str -> occursin(str, vn_str), expr_strs)
        return length(valid) > 0
    end
end

"""
`runmodel!(model::Model, vi::AbstractVarInfo, spl::AbstractSampler)`

Samples from `model` using the sampler `spl` storing the sample and log joint 
probability in `vi`.
"""
function runmodel!(model::Model, vi::AbstractVarInfo, spl::AbstractSampler = SampleFromPrior())
    setlogp!(vi, zero(Float64))
    if spl isa Sampler && haskey(spl.info, :eval_num)
        spl.info[:eval_num] += 1
    end
    model(vi, spl)
    return vi
end

VarInfo(meta=Metadata()) = VarInfo(meta, Ref{Real}(0.0), Ref(0))

"""
`TypedVarInfo(vi::UntypedVarInfo)`

This function finds all the unique `sym`s from the instances of `VarName{sym}` found in 
`vi.metadata.vns`. It then extracts the metadata associated with each symbol from the 
global `vi.metadata` field. Finally, a new `VarInfo` is created with a new `metadata` as 
a `NamedTuple` mapping from symbols to type-stable `Metadata` instances, one for each 
symbol.
"""
function TypedVarInfo(vi::UntypedVarInfo)
    meta = vi.metadata
    new_metas = Metadata[]
    # Symbols of all instances of `VarName{sym}` in `vi.vns`
    syms_tuple = Tuple(syms(vi))
    for s in syms_tuple
        # Find all indices in `vns` with symbol `s`
        inds = findall(vn -> vn.sym == s, vi.vns)
        n = length(inds)
        # New `vns`
        sym_vns = getindex.((vi.vns,), inds)
        # New idcs
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        # New dists
        sym_dists = getindex.((vi.dists,), inds)
        # New gids, can make a resizeable FillArray
        sym_gids = getindex.((vi.gids,), inds)
        @assert length(sym_gids) <= 1 || 
            all(x -> x == sym_gids[1], @view sym_gids[2:end])
        # New orders
        sym_orders = getindex.((vi.orders,), inds)
        # New flags
        sym_flags = Dict(a => vi.flags[a][inds] for a in keys(vi.flags))

        # Extract new ranges and vals
        _ranges = getindex.((vi.ranges,), inds)
        # `copy.()` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(vi.vals[_ranges[i]]) for i in 1:n]
        sym_ranges = Vector{eltype(_ranges)}(undef, n)
        start = 0
        for i in 1:n
            sym_ranges[i] = start + 1 : start + length(_vals[i])
            start += length(_vals[i])
        end
        sym_vals = foldl(vcat, _vals)

        push!(new_metas, Metadata(sym_idcs, sym_vns, sym_ranges, sym_vals, 
                                    sym_dists, sym_gids, sym_orders, sym_flags)
            )
    end
    logp = vi.logp
    num_produce = vi.num_produce
    nt = NamedTuple{syms_tuple}(Tuple(new_metas))
    return VarInfo(nt, Ref(logp), Ref(num_produce))
end

function getproperty(vi::VarInfo, f::Symbol)
    f === :logp && return getfield(vi, :logp)[]
    f === :num_produce && return getfield(vi, :num_produce)[]
    f === :metadata && return getfield(vi, :metadata)
    return getfield(getfield(vi, :metadata), f)
end
function setproperty!(vi::VarInfo, f::Symbol, x)
    f === :logp && return getfield(vi, :logp)[] = x
    f === :num_produce && return getfield(vi, :num_produce)[] = x
    return setfield!(vi, f, x)
end

"""
`empty!(vi::VarInfo)`

Empties all the fields of `vi.metadata` and resets `vi.logp` and `vi.num_produce` to 
zeros. This is useful when using a sampling algorithm that assumes an empty 
`vi::VarInfo`, e.g. `SMC`. 
"""
function empty!(vi::VarInfo)
    _empty!(vi.metadata)
    vi.logp = 0
    vi.num_produce = 0
    return vi
end
@inline _empty!(metadata::Metadata) = empty!(metadata)
@inline function _empty!(metadata::NamedTuple{names}) where {names}
    # Check if the named tuple is empty and end the recursion
    length(names) === 0 && return nothing
    # Take the first key in the NamedTuple
    f = names[1]
    # Empty the first instance of `Metadata`
    empty!(getfield(metadata, f))
    # Recurse using the remaining pairs of the NamedTuple
    return _empty!(_tail(metadata))
end

# Functions defined only for UntypedVarInfo
"""
`keys(vi::UntypedVarInfo)`

Returns an iterator over `vi.vns`.
"""
keys(vi::UntypedVarInfo) = keys(vi.idcs)

"""
`setgid!(vi::VarInfo, gid::Selector, vn::VarName)`

Adds `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
setgid!(vi::UntypedVarInfo, gid::Selector, vn::VarName) = push!(vi.gids[getidx(vi, vn)], gid)
function setgid!(vi::TypedVarInfo, gid::Selector, vn::VarName{sym}) where sym
    push!(getfield(vi.metadata, sym).gids[getidx(vi, vn)], gid)
end

"""
`istrans(vi::VarInfo, vn::VarName)`

Returns true if `vn`'s values in `vi` are transformed to Eucledian space, and false if 
they are in the support of `vn`'s distribution.
"""
istrans(vi::AbstractVarInfo, vn::VarName) = is_flagged(vi, vn, "trans")

"""
`getlogp(vi::VarInfo)`

Returns the log of the joint probability of the observed data and parameters sampled in 
`vi`.
"""
getlogp(vi::AbstractVarInfo) = vi.logp

"""
`setlogp!(vi::VarInfo, logp::Real)`

Sets the log of the joint probability of the observed data and parameters sampled in 
`vi` to `logp`.
"""
setlogp!(vi::AbstractVarInfo, logp::Real) = vi.logp = logp


"""
`acclogp!(vi::VarInfo, logp::Real)`

Adds `logp` to the value of the log of the joint probability of the observed data and 
parameters sampled in `vi`.
"""
acclogp!(vi::AbstractVarInfo, logp::Real) = vi.logp += logp

"""
`resetlogp!(vi::VarInfo)`

Resets the value of the log of the joint probability of the observed data and parameters 
sampled in `vi` to 0.
"""
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, 0.0)

"""
`isempty(vi::VarInfo)`

Returns true if `vi` is empty and false otherwise.
"""
isempty(vi::UntypedVarInfo) = isempty(vi.idcs)
isempty(vi::TypedVarInfo) = _isempty(vi.metadata)
@inline function _isempty(metadata::NamedTuple{names}) where {names}
    # Checks if `metadata` is empty to end the recursion
    length(names) === 0 && return true
    # Take the first key of `metadata`
    f = names[1]
    # If not empty, return false and end the recursion. Otherwise, recurse using the remaining of `metadata`.
    return isempty(getfield(metadata, f).idcs) && _isempty(_tail(metadata))
end

# X -> R for all variables associated with given sampler
"""
`link!(vi::VarInfo, spl::Sampler)`

Transforms the values of the random variables sampled by `spl` in `vi` from the support 
of their distributions to the Eucledian space and sets their corresponding ``"trans"` 
flag values to `true`.
"""
function link!(vi::UntypedVarInfo, spl::Sampler)
    # TODO: Change to a lazy iterator over `vns`
    vns = _getvns(vi, spl)
    if ~istrans(vi, vns[1])
        for vn in vns
            dist = getdist(vi, vn)
            # TODO: Use inplace versions to avoid allocations
            setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
            settrans!(vi, true, vn)
        end
    else
        @warn("[Turing] attempt to link a linked vi")
    end
end
function link!(vi::TypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    space = getspace(spl)
    return _link!(vi.metadata, vi, vns, space)
end
@inline function _link!(metadata::NamedTuple{names}, vi, vns, space) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first key/symbol of `metadata`
    f = names[1]
    # Extract the list of `vns` with symbol `f`
    f_vns = getfield(vns, f)
    # Transform only if `f` is in the space of the sampler or the space is void
    if f ∈ space || length(space) == 0
        if ~istrans(vi, f_vns[1])
            # Iterate over all `f_vns` and transform
            for vn in f_vns
                dist = getdist(vi, vn)
                setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
                settrans!(vi, true, vn)
            end
        else
            @warn("[Turing] attempt to link a linked vi")
        end
    end
    # Recurse using the remaining of `metadata`
    return _link!(_tail(metadata), vi, vns, space)
end

# R -> X for all variables associated with given sampler
"""
`invlink!(vi::VarInfo, spl::Sampler)`

Transforms the values of the random variables sampled by `spl` in `vi` from the 
Eucledian space back to the support of their distributions and sets their corresponding 
``"trans"` flag values to `false`.
"""
function invlink!(vi::UntypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
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
function invlink!(vi::TypedVarInfo, spl::Sampler)
    vns = _getvns(vi, spl)
    space = getspace(spl)
    return _invlink!(vi.metadata, vi, vns, space)
end
@inline function _invlink!(metadata::NamedTuple{names}, vi, vns, space) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first key/symbol of `metadata`
    f = names[1]
    # Extract the list of `vns` with symbol `f`
    f_vns = getfield(vns, f)
    # Transform only if `f` is in the space of the sampler or the space is void
    if f ∈ space || length(space) == 0
        if istrans(vi, f_vns[1])
            # Iterate over all `f_vns` and transform
            for vn in f_vns
                dist = getdist(vi, vn)
                setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
                settrans!(vi, false, vn)
            end
        else
            @warn("[Turing] attempt to invlink an invlinked vi")
        end
    end
    return _invlink!(_tail(metadata), vi, vns, space)
end

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
"""
`getindex(vi::VarInfo, vn::VarName)`
`getindex(vi::VarInfo, vns::Vector{<:VarName})`

Returns the current value(s) of `vn` (`vns`) in `vi` in the support of its (their) 
distribution(s). If the value(s) is (are) transformed to the Eucledian space, it is 
(they are) transformed back.
"""
function getindex(vi::AbstractVarInfo, vn::VarName)
    @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vn)
    return copy(istrans(vi, vn) ?
        invlink(dist, reconstruct(dist, getval(vi, vn))) :
        reconstruct(dist, getval(vi, vn)))
end
function getindex(vi::AbstractVarInfo, vns::Vector{<:VarName})
    @assert haskey(vi, vns[1]) "[Turing] attempted to replay unexisting variables in VarInfo"
    dist = getdist(vi, vns[1])
    return copy(istrans(vi, vns[1]) ?
        invlink(dist, reconstruct(dist, getval(vi, vns), length(vns))) :
        reconstruct(dist, getval(vi, vns), length(vns)))
end

"""
`getindex(vi::VarInfo, spl::Union{SampleFromPrior, Sampler})`

Returns the current value(s) of the random variables sampled by `spl` in `vi`. The 
value(s) may or may not be transformed to Eucledian space.
"""
getindex(vi::AbstractVarInfo, spl::SampleFromPrior) = copy(getall(vi))
getindex(vi::UntypedVarInfo, spl::Sampler) = copy(getval(vi, _getranges(vi, spl)))
function getindex(vi::TypedVarInfo, spl::Sampler)
    # Gets the ranges as a NamedTuple
    ranges = _getranges(vi, spl)
    # Calling getfield(ranges, f) gives all the indices in `vals` of the `vn`s with symbol `f` sampled by `spl` in `vi`
    return vcat(_getindex(vi.metadata, ranges)...)
end
# Recursively builds a tuple of the `vals` of all the symbols
@inline function _getindex(metadata::NamedTuple{names}, ranges) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return ()
    # Take the first key of `metadata`
    f = names[1]
    # Get the `vals` and `ranges` of symbol `f`
    f_vals = getfield(metadata, f).vals
    f_range = getfield(ranges, f)
    # Get the values from `f_vals` that were sampled by `spl` and recurse using the remaining of `metadata` 
    return (f_vals[f_range], _getindex(_tail(metadata), ranges)...)
end

"""
`setindex!(vi::VarInfo, val, vn::VarName)`

Sets the current value(s) of the random variable `vn` in `vi` to `val`. The value(s) may 
or may not be transformed to Eucledian space.
"""
setindex!(vi::AbstractVarInfo, val::Any, vn::VarName) = setval!(vi, val, vn)

"""
`setindex!(vi::VarInfo, val, spl::Union{SampleFromPrior, Sampler})`

Sets the current value(s) of the random variables sampled by `spl` in `vi` to `val`. The 
value(s) may or may not be transformed to Eucledian space.
"""
setindex!(vi::AbstractVarInfo, val::Any, spl::SampleFromPrior) = setall!(vi, val)
setindex!(vi::UntypedVarInfo, val::Any, spl::Sampler) = setval!(vi, val, _getranges(vi, spl))
function setindex!(vi::TypedVarInfo, val, spl::Sampler)
    # Gets a `NamedTuple` mapping each symbol to the indices in the symbol's `vals` field sampled from the sampler `spl`
    ranges = _getranges(vi, spl)
    _setindex!(vi.metadata, val, ranges)
    return val
end
# Recursively writes the entries of `val` to the `vals` fields of all the symbols as if they were a contiguous vector.
@inline function _setindex!(metadata::NamedTuple{names}, val, ranges, start = 0) where {names}
    length(names) === 0 && return nothing
    f = names[1]
    # The `vals` field of symbol `f`
    f_vals = getfield(metadata, f).vals
    # The indices in `f_vals` corresponding to sampler `spl`
    f_range = getfield(ranges, f)
    n = length(f_range)
    # Writes the portion of `val` corresponding to the symbol `f`
    @views f_vals[f_range] .= val[start+1:start+n]
    # Increment the global index and move to the next symbol
    start += n
    return _setindex!(_tail(metadata), val, ranges, start)
end

"""
`haskey(vi::VarInfo, vn::VarName)`

Returns `true` if `vn` has been sampled in `vi` and `false` otherwise.
"""
haskey(vi::UntypedVarInfo, vn::VarName) = haskey(vi.idcs, vn)
function haskey(vi::TypedVarInfo, vn::VarName{sym}) where {sym}
    metadata = vi.metadata
    Tmeta = typeof(metadata)
    return sym in fieldnames(Tmeta) && haskey(getfield(metadata, sym).idcs, vn)
end

function show(io::IO, vi::UntypedVarInfo)
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
"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution)`

Pushes a new random variable `vn` with a sampled value `r` from a distribution `dist` to 
the `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution)
    return push!(vi, vn, r, dist, Set{Selector}([]))
end

"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution, spl::AbstractSampler)`

Pushes a new random variable `vn` with a sampled value `r` sampled with a sampler `spl` 
from a distribution `dist` to `VarInfo` `vi`. The sampler is passed here to invalidate 
its cache where defined.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, spl::Sampler)
    spl.info[:cache_updated] = CACHERESET
    return push!(vi, vn, r, dist, spl.selector)
end
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, spl::AbstractSampler)
    return push!(vi, vn, r, dist)
end

"""
`push!(vi::VarInfo, vn::VarName, r, dist::Distribution, gid::Selector)`

Pushes a new random variable `vn` with a sampled value `r` sampled with a sampler of 
selector `gid` from a distribution `dist` to `VarInfo` `vi`.
"""
function push!(vi::AbstractVarInfo, vn::VarName, r::Any, dist::Distribution, gid::Selector)
    return push!(vi, vn, r, dist, Set([gid]))
end
function push!(vi::UntypedVarInfo, vn::VarName, r::Any, dist::Distribution, gidset::Set{Selector})
    @assert ~(vn in keys(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

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
            vi::TypedVarInfo, 
            vn::VarName{sym}, 
            r::Any, 
            dist::Distributions.Distribution, 
            gidset::Set{Selector}
            ) where sym
    
    @assert ~(haskey(vi, vn)) "[push!] attempt to add an exisitng variable $(vn.sym) ($(vn)) to TypedVarInfo of syms $(syms(vi)) with dist=$dist, gid=$gid"

    val = vectorize(dist, r)

    meta = getfield(vi.metadata, sym)
    meta.idcs[vn] = length(meta.idcs) + 1
    push!(meta.vns, vn)
    l = length(meta.vals); n = length(val)
    push!(meta.ranges, l+1:l+n)
    append!(meta.vals, val)
    push!(meta.dists, dist)
    push!(meta.gids, gidset)
    push!(meta.orders, vi.num_produce)
    push!(meta.flags["del"], false)
    push!(meta.flags["trans"], false)

    return meta
end

"""
`setorder!(vi::VarInfo, vn::VarName, index::Int)`

Sets the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe 
statements run before sampling `vn`.
"""
function setorder!(vi::UntypedVarInfo, vn::VarName, index::Int)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    return vi
end
function setorder!(mvi::TypedVarInfo, vn::VarName{sym}, index::Int) where {sym}
    vi = getfield(mvi.metadata, sym)
    if vi.orders[vi.idcs[vn]] != index
        vi.orders[vi.idcs[vn]] = index
    end
    return mvi
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

"""
`is_flagged(vi::VarInfo, vn::VarName, flag::String)`

Returns `true` if `vn` has a true value for `flag` in `vi`, and `false` otherwise.
"""
function is_flagged(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)]
end
function is_flagged(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)]
end

"""
`unset_flag!(vi::VarInfo, vn::VarName, flag::String)`

Sets `vn`'s value for `flag` to `false` in `vi`.
"""
function unset_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)] = false
end
function unset_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)] = false
end

"""
`set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)`

Sets the `"del"` flag of variables in `vi` with `order > vi.num_produce` to `true`.
"""
function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = _getidcs(vi, spl)
    if vi.num_produce == 0
        for i = length(gidcs):-1:1
          vi.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in gidcs && vi.orders[i] > vi.num_produce
                vi.flags["del"][i] = true
            end
        end
    end
end
function set_retained_vns_del_by_spl!(vi::TypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = _getidcs(vi, spl)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, vi.num_produce)
end
@inline function _set_retained_vns_del_by_spl!(metadata::NamedTuple{names}, gidcs, num_produce) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return nothing
    # Take the first symbol
    f = names[1]
    # Get the idcs, orders and flags with symbol `f`
    f_gidcs = getfield(gidcs, f)
    f_orders = getfield(metadata, f).orders
    f_flags = getfield(metadata, f).flags
    # Set the flag for variables with symbol `f`
    if num_produce == 0
        for i = length(f_gidcs):-1:1
            f_flags["del"][f_gidcs[i]] = true
        end
    else
        for i in 1:length(f_orders)
            if i in f_gidcs && f_orders[i] > num_produce
                f_flags["del"][i] = true
            end
        end
    end
    # Recurse using the remaining of `metadata`
    return _set_retained_vns_del_by_spl!(_tail(metadata), gidcs, num_produce)
end

"""
`updategid!(vi::VarInfo, vn::VarName, spl::Sampler)`

If `vn` doesn't have a sampler selector linked and `vn`'s symbol is in the space of 
`spl`, this function will set `vn`'s `gid` to `Set([spl.selector])`.
"""
function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler)
    if ~isempty(spl.alg.space) && isempty(getgid(vi, vn)) && getsym(vn) in spl.alg.space
        setgid!(vi, spl.selector, vn)
    end
end
