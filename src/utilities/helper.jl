struct FlattenIterator{Tname, Tvalue}
    name::Tname
    value::Tvalue
end

Base.length(iter::FlattenIterator) = _length(iter.value)
_length(a::AbstractArray) = sum(_length, a)
_length(a::AbstractArray{<:Number}) = length(a)
_length(::Number) = 1

Base.eltype(iter::FlattenIterator{String}) = Tuple{String, _eltype(typeof(iter.value))}
_eltype(::Type{TA}) where {TA <: AbstractArray} = _eltype(eltype(TA))
_eltype(::Type{T}) where {T <: Number} = T

@inline function Base.iterate(iter::FlattenIterator{String, <:Number}, i = 1)
    i === 1 && return (iter.name, iter.value), 2
    return nothing
end
@inline function Base.iterate(
    iter::FlattenIterator{String, <:AbstractArray{<:Number}}, 
    ind = (1,),
)
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    return (name, iter.value[i]), (i+1,)
end
@inline function Base.iterate(
    iter::FlattenIterator{String, T},
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
    name = string(ind2sub(size(iter.value), i))
    name = replace(name, "(" => "[");
    name = replace(name, ",)" => "]");
    name = replace(name, ")" => "]");
    name = iter.name * name
    return name
end
@inline ind2sub(v, i) = Tuple(CartesianIndices(v)[i])
