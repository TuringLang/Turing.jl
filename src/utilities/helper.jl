struct FlattenIterator{Tname, Tvalue}
    name::Tname
    value::Tvalue
end

Base.length(iter::FlattenIterator) where {T} = _length(iter.value)
@inline _length(a::AbstractArray) = sum(_length, a)
@inline _length(::Number) = 1

@inline function Base.iterate(iter::FlattenIterator{String, <:Number}, i = 1)
    i === 1 && return (iter.name, iter.value), 2
    return nothing
end
@inline function Base.iterate(iter::FlattenIterator{String, T}, ind = (1,)) where {T <: AbstractArray{<:Number}}
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    return (name, iter.value[i]), (i+1,)
end
@inline function Base.iterate(iter::FlattenIterator{String, T}, ind = startind(T)) where {T <: AbstractArray}
    @show ind
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

startind(::Type{<:AbstractArray{T}}) where {T} = (1, startind(T)...)
startind(::Type{<:Number}) = ()
startind(::Type{<:Any}) = throw("Type not supported.")
function getname(iter::FlattenIterator, i::Int)
    name = string(ind2sub(size(iter.value), i))
    name = replace(name, "(" => "[");
    name = replace(name, ",)" => "]");
    name = replace(name, ")" => "]");
    name = iter.name * name
    return name
end
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])
