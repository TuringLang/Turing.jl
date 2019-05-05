"""
`Metadata()`

Constructs an empty type unstable instance of `Metadata`.
"""
function Metadata()
    vals  = Vector{Real}()
    flags = Dict{String, BitVector}()
    flags["del"] = BitVector()
    flags["trans"] = BitVector()

    return Metadata(
        Dict{VarName, Int}(),
        Vector{VarName}(),
        Vector{UnitRange{Int}}(),
        vals,
        Vector{Distributions.Distribution}(),
        Vector{Set{Selector}}(),
        Vector{Int}(),
        flags
    )
end

"""
`empty!(meta::Metadata)`

Empties all the fields of `meta`. This is useful when using a sampling algorithm that 
assumes an empty `meta`, e.g. `SMC`. 
"""
function empty!(meta::Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end

    return meta
end

# Removes the first element of a NamedTuple. The pairs in a NamedTuple are ordered, so this is well-defined.
if VERSION < v"1.1"
    _tail(nt::NamedTuple{names}) where names = NamedTuple{Base.tail(names)}(nt)
else
    _tail(nt::NamedTuple) = Base.tail(nt)
end

const VarView = Union{Int, UnitRange, Vector{Int}}

"""
`getval(vi::UntypedVarInfo, vview::Union{Int, UnitRange, Vector{Int}})`

Returns a view `vi.vals[vview]`.
"""
getval(vi::UntypedVarInfo, vview::VarView) = view(vi.vals, vview)

"""
`setval!(vi::UntypedVarInfo, val, vview::Union{Int, UnitRange, Vector{Int}})`

Sets the value of `vi.vals[vview]` to `val`.
"""
setval!(vi::UntypedVarInfo, val, vview::VarView) = vi.vals[vview] = val
function setval!(vi::UntypedVarInfo, val, vview::Vector{UnitRange})
    if length(vview) > 0
        vi.vals[[i for arr in vview for i in arr]] = val
    end
    return val
end

"""
`getidx(vi::UntypedVarInfo, vn::VarName)`

Returns the index of `vn` in `vi.metadata.vns`.
"""
getidx(vi::UntypedVarInfo, vn::VarName) = vi.idcs[vn]

"""
`getidx(vi::TypedVarInfo, vn::VarName{sym})`

Returns the index of `vn` in `getfield(vi.metadata, sym).vns`.
"""
function getidx(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).idcs[vn]
end

"""
`getrange(vi::UntypedVarInfo, vn::VarName)`

Returns the index range of `vn` in `vi.metadata.vals`.
"""
getrange(vi::UntypedVarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]

"""
`getrange(vi::TypedVarInfo, vn::VarName{sym})`

Returns the index range of `vn` in `getfield(vi.metadata, sym).vals`.
"""
function getrange(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).ranges[getidx(vi, vn)]
end

"""
`getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})`

Returns all the indices of `vns` in `vi.metadata.vals`.
"""
function getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return union(map(vn -> getrange(vi, vn), vns)...)
end

"""
`getdist(vi::VarInfo, vn::VarName)`

Returns the distribution from which `vn` was sampled in `vi`.
"""
getdist(vi::UntypedVarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]
function getdist(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).dists[getidx(vi, vn)]
end

"""
`getval(vi::VarInfo, vn::VarName)`

Returns the value(s) of `vn`. The values may or may not be transformed to Eucledian space.
"""
getval(vi::UntypedVarInfo, vn::VarName) = view(vi.vals, getrange(vi, vn))
function getval(vi::TypedVarInfo, vn::VarName{sym}) where sym
    view(getfield(vi.metadata, sym).vals, getrange(vi, vn))
end

"""
`setval!(vi::VarInfo, val, vn::VarName)`

Sets the value(s) of `vn` in `vi.metadata` to `val`. The values may or may not be 
transformed to Eucledian space.
"""
setval!(vi::UntypedVarInfo, val, vn::VarName) = vi.vals[getrange(vi, vn)] = val
function setval!(vi::TypedVarInfo, val, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).vals[getrange(vi, vn)] = val
end

"""
`getval(vi::VarInfo, vns::Vector{<:VarName})`

Returns all the value(s) of `vns`. The values may or may not be transformed to Eucledian 
space.
"""
getval(vi::UntypedVarInfo, vns::Vector{<:VarName}) = view(vi.vals, getranges(vi, vns))
function getval(vi::TypedVarInfo, vns::Vector{VarName{sym}}) where sym
    view(getfield(vi.metadata, sym).vals, getranges(vi, vns))
end

"""
`getall(vi::VarInfo)`

Returns the values of all the variables in `vi`. The values may or may not be 
transformed to Eucledian space.
"""
getall(vi::UntypedVarInfo) = vi.vals
getall(vi::TypedVarInfo) = vcat(_getall(vi.metadata)...)
@inline function _getall(metadata::NamedTuple{names}) where {names}
    # Check if NamedTuple is empty and end recursion
    length(names) === 0 && return ()
    # Take the first key of the NamedTuple
    f = names[1]
    # Recurse using the remaining of `metadata`
    return (getfield(metadata, f).vals, _getall(_tail(metadata))...)
end

"""
`setall!(vi::VarInfo, val)`

Sets the values of all the variables in `vi` to `val`. The values may or may not be 
transformed to Eucledian space.
"""
setall!(vi::UntypedVarInfo, val) = vi.vals .= val
setall!(vi::TypedVarInfo, val) = _setall!(vi.metadata, val)
@inline function _setall!(metadata::NamedTuple{names}, val, start = 0) where {names}
    # Check if `metadata` is empty and end recursion
    length(names) === 0 && return nothing
    # Take the first key of `metadata`
    f = names[1]
    # Set the `vals` of the current symbol, i.e. f, to the relevant portion in `val`
    vals = getfield(metadata, f).vals
    @views vals .= val[start + 1 : start + length(vals)]
    # Recurse using the remaining of `metadata` and the remaining of `val`
    return _setall(_tail(metadata), val, start + length(vals))
end

"""
`getsym(vn::VarName)`

Returns the symbol of the Julia variable used to generate `vn`.
"""
getsym(vn::VarName{sym}) where sym = sym

"""
`getgid(vi::VarInfo, vn::VarName)`

Returns the set of sampler selectors associated with `vn` in `vi`.
"""
getgid(vi::UntypedVarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]
function getgid(vi::TypedVarInfo, vn::VarName{sym}) where sym
    getfield(vi.metadata, sym).gids[getidx(vi, vn)]
end

"""
`settrans!(vi::VarInfo, trans::Bool, vn::VarName)`

Sets the `trans` flag value of `vn` in `vi`.
"""
function settrans!(vi::AbstractVarInfo, trans::Bool, vn::VarName)
    trans ? set_flag!(vi, vn, "trans") : unset_flag!(vi, vn, "trans")
end

"""
`syms(vi::VarInfo)`

Returns a tuple of the unique symbols of random variables sampled in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(vn -> vn.sym, vi.vns)))  # get all symbols
syms(vi::TypedVarInfo) = fieldnames(vi.metadata)

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
function _getidcs(vi::UntypedVarInfo, ::SampleFromPrior)
    return filter(i -> isempty(vi.gids[i]) , 1:length(vi.gids))
end
# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
function _getidcs(vi::TypedVarInfo, ::SampleFromPrior)
    return __getidcs(vi.metadata)
end
@inline function __getidcs(metadata::NamedTuple{names}) where {names}
    # Check if the `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first key/symbol
    f = names[1]
    # Get the first symbol's metadata
    meta = getfield(metadata, f)
    # Get all the idcs of vns with empty gid
    v = filter(i -> isempty(meta.gids[i]), 1:length(meta.gids))
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of metadata
    return merge(nt, __getidcs(_tail(metadata)))
end

# Get all indices of variables belonging to a given sampler
function _getidcs(vi::AbstractVarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    # Checks if cache is valid, i.e. no new pushes were made, to return the cached idcs
    # Otherwise, it recomputes the idcs and caches it
    if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
        spl.info[:idcs]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
        spl.info[:idcs] = _getidcs(vi, spl.selector, spl.alg.space)
    end
end
function _getidcs(vi::UntypedVarInfo, s::Selector, space)
    filter(i -> (s in vi.gids[i] || isempty(vi.gids[i])) && 
        (isempty(space) || in(vi.vns[i], space)), 1:length(vi.gids))
end
function _getidcs(vi::TypedVarInfo, s::Selector, space)
    return __getidcs(vi.metadata, s, space)
end
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@inline function __getidcs(metadata::NamedTuple{names}, s::Selector, space) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first sybmol
    f = names[1]
    # Get the first symbol's metadata
    f_meta = getfield(metadata, f)
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    v = filter((i) -> (s in f_meta.gids[i] || isempty(f_meta.gids[i])) && 
        (isempty(space) || in(f_meta.vns[i], space)), 1:length(f_meta.gids))
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of metadata
    return merge(nt, __getidcs(_tail(metadata), s, space))
end

# Get all vns of variables belonging to spl
_getvns(vi::UntypedVarInfo, spl::AbstractSampler) = view(vi.vns, _getidcs(vi, spl))
function _getvns(vi::TypedVarInfo, spl::AbstractSampler) 
    # Get a NamedTuple of the indices of variables belonging to `spl`, one entry for each symbol
    idcs = _getidcs(vi, spl)
    return __getvns(vi.metadata, idcs)
end
# Get a NamedTuple for all the `vns` of indices `idcs`, one entry for each symbol
@inline function __getvns(metadata::NamedTuple{names}, idcs) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first symbol
    f = names[1]
    # Get the vector of `vns` with symbol `f`
    v = getfield(metadata, f).vns[getfield(idcs, f)]
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of `metadata`
    return merge(nt, __getvns(_tail(metadata), idcs))
end

# Get the index (in vals) ranges of all the vns of variables belonging to spl
function _getranges(vi::AbstractVarInfo, spl::Sampler)
    if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
        spl.info[:ranges]
    else
        spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        spl.info[:ranges] = _getranges(vi, spl.selector, spl.alg.space)
    end
end
# Get the index (in vals) ranges of all the vns of variables belonging to selector `s` in `space`
function _getranges(vi::AbstractVarInfo, s::Selector, space::Set=Set())
    __getranges(vi, _getidcs(vi, s, space))
end
function __getranges(vi::UntypedVarInfo, idcs)
    union(map(i -> vi.ranges[i], idcs)...)
end
__getranges(vi::TypedVarInfo, idcs) = __getranges(vi.metadata, idcs)
@inline function __getranges(metadata::NamedTuple{names}, idcs) where {names}
    # Check if `metadata` is empty to end the recursion
    length(names) === 0 && return NamedTuple()
    # Take the first symbol
    f = names[1]
    # Collect the index ranges of all the vns with symbol `f` 
    v = union(map(i -> getfield(metadata, f).ranges[i], getfield(idcs, f))..., Int[])
    # Make a single-pair NamedTuple to merge with the result of the recursion
    nt = NamedTuple{(f,)}((v,))
    # Recurse using the remaining of `metadata`
    return merge(nt, __getranges(_tail(metadata), idcs))
end

"""
`set_flag!(vi::VarInfo, vn::VarName, flag::String)`

Sets `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::UntypedVarInfo, vn::VarName, flag::String)
    return vi.flags[flag][getidx(vi, vn)] = true
end
function set_flag!(vi::TypedVarInfo, vn::VarName{sym}, flag::String) where {sym}
    return getfield(vi.metadata, sym).flags[flag][getidx(vi, vn)] = true
end
