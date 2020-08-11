struct FlattenIterator{T}
    name::Symbol
    value::T
end

FlattenIterator(name, value) = FlattenIterator(Symbol(name), value)

Base.length(iter::FlattenIterator) = _length(iter.value)
_length(a) = length(a)
_length(a::AbstractArray) = sum(_length, a)

Base.eltype(::Type{FlattenIterator{T}}) where T = Tuple{Symbol,_eltype(T)}
_eltype(::Type{T}) where T = eltype(T)
_eltype(::Type{TA}) where {TA<:AbstractArray} = _eltype(eltype(TA))

@inline function Base.iterate(iter::FlattenIterator{<:Number}, i = 1)
    i === 1 && return (iter.name, iter.value), 2
    return nothing
end
@inline function Base.iterate(
    iter::FlattenIterator{<:AbstractArray{<:Number}},
    ind = (1,),
)
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    return (name, iter.value[i]), (i+1,)
end
@inline function Base.iterate(
    iter::FlattenIterator{T},
    ind = startind(T),
) where {T <: AbstractArray}
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    local out
    while i <= length(iter.value)
        v = iter.value[i]
        out = iterate(FlattenIterator(name, v), Base.tail(ind))
        out !== nothing && break
        i += 1
    end
    if out === nothing
        return nothing
    else
        return out[1], (i, out[2]...)
    end
end

@inline startind(::Type{<:AbstractArray{T}}) where {T} = (1, startind(T)...)
@inline startind(::Type{<:Number}) = ()
@inline startind(::Type{<:Any}) = throw("Type not supported.")
@inline function getname(iter::FlattenIterator, i::Int)
    return Symbol(iter.name, "[", join(ind2sub(size(iter.value), i), ","), "]")
end
@inline ind2sub(v, i) = Tuple(CartesianIndices(v)[i])
