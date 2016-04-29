# Implementation of data structures that automatically
#  perform copy-on-write after task copying.

# If current_task is an existing key in s, then return s[current_task].
# Otherwise, return s[current_task] = s[last_task].
immutable TArray{T,N} <: DenseArray{T,N}
  ref :: Symbol  # object_id
  TArray() = new(gensym())
end

call{T}(::Type{TArray{T,1}}, d::Integer)  = TArray(T,  d)
call{T}(::Type{TArray{T}}, d::Integer...) = TArray(T, convert(Tuple{Vararg{Int}}, d))
call{T,N}(::Type{TArray{T,N}}, d::Integer...) = length(d)==N ? TArray(T, convert(Tuple{Vararg{Int}}, d)) : error("malformed dims")
call{T,N}(::Type{TArray{T,N}}, dim::NTuple{N,Int}) = TArray(T, dim)

function TArray(T::Type, dim)
  res = TArray{T,length(dim)}();
  t = current_task()
  d = Array{T}(dim)
  task_local_storage(res.ref, (t,d))
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
  t, d = task_local_storage(S.ref)
  ct = current_task()
  newd = d
  if t != ct
    # println("[setindex!]: $(S.ref) copying data")
    newd = deepcopy(d)
    task_local_storage(S.ref, (ct, newd))
  end
  setindex!(newd, x, i)
end

function Base.push!(S::TArray, x)
  t, d = task_local_storage(S.ref)
  ct = current_task()
  newd = d
  if t != ct
    newd = deepcopy(d)
    task_local_storage(S.ref, (ct, newd))
  end
  push!(newd, x)
end

function Base.pop!(S::TArray)
  t, d = task_local_storage(S.ref)
  ct = current_task()
  newd = d
  if t != ct
    newd = deepcopy(d)
    task_local_storage(S.ref, (ct, newd))
  end
  pop!(d)
end

function Base.convert(::Type{TArray}, x::Array)
  res = TArray{typeof(x[1]),ndims(x)}();
  t = current_task()
  task_local_storage(res.ref, (t,x))
  res
end

Base.show(io::IO, S::TArray) = Base.show(io::IO, task_local_storage(S.ref)[2])
Base.display(io::IO, S::TArray) = Base.display(io::IO, task_local_storage(S.ref)[2])
Base.size(S::TArray) = Base.size(task_local_storage(S.ref)[2])
Base.ndims(S::TArray) = Base.ndims(task_local_storage(S.ref)[2])

Base.get(t::Task, S) = S
Base.get(t::Task, S::TArray) = (t.storage[S.ref][2])

## convenience constructors ##

"""
     tzeros(dims, ...)
Construct a distributed array of zeros.
Trailing arguments are the same as those accepted by `TArray`.
"""
function tzeros(T::Type, dim)
  res = TArray{T,length(dim)}();
  t = current_task()
  d = zeros(T,dim)
  task_local_storage(res.ref, (t,d))
  res
end
tzeros{T}(::Type{T}, d1::Integer, drest::Integer...) = tzeros(T, convert(Dims, tuple(d1, drest...)))
tzeros(d1::Integer, drest::Integer...) = tzeros(Float64, convert(Dims, tuple(d1, drest...)))
tzeros(d::Dims) = tzeros(Float64, d)
