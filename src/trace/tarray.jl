##########
# TArray #
##########

using Markdown

doc"""
    TArray{T}(dims, ...)

Implementation of data structures that automatically perform copy-on-write after task copying.

If current_task is an existing key in `s`, then return `s[current_task]`. Otherwise, return `s[current_task] = s[last_task]`.

Usage:

```julia
TArray(dim)
```

Example:

```julia
ta = TArray(4)              # init
for i in 1:4 ta[i] = i end  # assign
Array(ta)                   # convert to 4-element Array{Int64,1}: [1, 2, 3, 4]
```
"""
struct TArray{T,N} <: DenseArray{T,N}
  ref :: Symbol  # object_id
  TArray{T,N}() where {T,N} = new(gensym())
end

TArray{T}() where T = TArray(T,  d)
TArray{T,1}(d::Integer) where T = TArray(T,  d)
TArray{T}(d::Integer...) where T = TArray(T, convert(Tuple{Vararg{Int}}, d))
TArray{T,N}(d::Integer...) where {T,N} = length(d)==N ? TArray(T, convert(Tuple{Vararg{Int}}, d)) : error("malformed dims")
TArray{T,N}(dim::NTuple{N,Int}) where {T,N} = TArray(T, dim)

function TArray(T::Type, dim)
  res = TArray{T,length(dim)}();
  n = n_copies()
  d = Array{T}(dim)
  task_local_storage(res.ref, (n,d))
  res
end

# pass through getindex and setindex!
# duplicate TArray if task id does not match current_task
function Base.getindex(S::TArray, i::Real)
  t, d = task_local_storage(S.ref)
  newd = d
#   ct = current_task()
#   if t != ct
#     # println("[getindex]: $(S.ref ) copying data")
#     newd = deepcopy(d)
#     task_local_storage(S.ref, (ct, newd))
#   end
  getindex(newd, i)
end

function Base.setindex!(S::TArray, x, i::Real)
  n, d = task_local_storage(S.ref)
  cn   = n_copies()
  newd = d
  if cn > n
    # println("[setindex!]: $(S.ref) copying data")
    newd = deepcopy(d)
    task_local_storage(S.ref, (cn, newd))
  end
  setindex!(newd, x, i)
end

function Base.push!(S::TArray, x)
  n, d = task_local_storage(S.ref)
  cn   = n_copies()
  newd = d
  if cn > n
    newd = deepcopy(d)
    task_local_storage(S.ref, (cn, newd))
  end
  push!(newd, x)
end

function Base.pop!(S::TArray)
  n, d = task_local_storage(S.ref)
  cn   = n_copies()
  newd = d
  if cn > n
    newd = deepcopy(d)
    task_local_storage(S.ref, (cn, newd))
  end
  pop!(d)
end

function Base.convert(::Type{TArray}, x::Array)
  res = TArray{typeof(x[1]),ndims(x)}();
  n   = n_copies()
  task_local_storage(res.ref, (n,x))
  res
end

function Base.convert(::Array, x::Type{TArray})
  n,d = task_local_storage(S.ref)
  c = deepcopy(d)
  return c
end


Base.show(io::IO, S::TArray) = Base.show(io::IO, task_local_storage(S.ref)[2])
Base.size(S::TArray) = Base.size(task_local_storage(S.ref)[2])
Base.ndims(S::TArray) = Base.ndims(task_local_storage(S.ref)[2])

# Base.get(t::Task, S) = S
# Base.get(t::Task, S::TArray) = (t.storage[S.ref][2])
Base.get(S::TArray) = (current_task().storage[S.ref][2])


##########
# tzeros #
##########

doc"""
     tzeros(dims, ...)

Construct a distributed array of zeros.
Trailing arguments are the same as those accepted by `TArray`.

```julia
tzeros(dim)
```

Example:

```julia
tz = tzeros(4)              # construct
Array(tz)                   # convert to 4-element Array{Int64,1}: [0, 0, 0, 0]
```
"""
function tzeros(T::Type, dim)
  res = TArray{T,length(dim)}();
  n = n_copies()
  d = zeros(T,dim)
  task_local_storage(res.ref, (n,d))
  res
end

tzeros(::Type{T}, d1::Integer, drest::Integer...) where T = tzeros(T, convert(Dims, tuple(d1, drest...)))
tzeros(d1::Integer, drest::Integer...) = tzeros(Float64, convert(Dims, tuple(d1, drest...)))
tzeros(d::Dims) = tzeros(Float64, d)
