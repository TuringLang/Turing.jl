using Turing: check_deps, libtask

check_deps()
libpath = dirname(libtask)

if !(libpath in Base.DL_LOAD_PATH)
  push!(Base.DL_LOAD_PATH, libpath)
end
# println(libpath)

# Utility function for self-copying mechanism
n_copies() = n_copies(current_task())
n_copies(t::Task) = begin
  isa(t.storage, Nothing) && (t.storage = IdDict())
  if haskey(t.storage, :n_copies)
    t.storage[:n_copies]
  else
    t.storage[:n_copies] = 0
  end
end


function Base.copy(t::Task)
  t.state != :runnable && t.state != :done &&
    error("Only runnable or finished tasks can be copied.")
  newt = ccall((:jl_clone_task, "libtask"), Any, (Any,), t)::Task
  if t.storage != nothing
    n = n_copies(t)
    t.storage[:n_copies]  = 1 + n
    newt.storage = copy(t.storage)
    #c = Channel(0) # new channel for copied task
    #newt.storage[:__chn__] = c
    #push!(c.putters, newt)
  else
    newt.storage = nothing
  end
  newt.code = t.code
  newt.state = t.state
  newt.result = t.result
  newt.parent = t.parent
  if :last in fieldnames(typeof(t))
    newt.last = nothing
  end
  newt
end

consume(t) = begin
    if t.storage == nothing
        t.storage = IdDict()
        t.storage[:__chn__] = Channel(0)
        # schedule(t)
    end
    take!(t.storage[:__chn__])
end
produce(x) = begin
    ct  = current_task()
    if ct.storage == nothing
        ct.storage = IdDict()
        ct.storage[:__chn__] = Channel(0)
    end
    println(ct)
    chn = ct.storage[:__chn__]
    put!(chn, x);
end
