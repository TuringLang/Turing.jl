module Traces
using Markdown
using Turing: Sampler
using Turing.VarReplay

# Trick for supressing some warning messages.
#   URL: https://github.com/KristofferC/OhMyREPL.jl/issues/14#issuecomment-242886953
macro suppress_err(block)
    quote
        if ccall(:jl_generating_output, Cint, ()) == 0
            ORIGINAL_STDERR = STDERR
            err_rd, err_wr = redirect_stderr()

            value = $(esc(block))

            REDIRECTED_STDERR = STDERR
            # need to keep the return value live
            err_stream = redirect_stderr(ORIGINAL_STDERR)

            return value
        end
    end
end

include("taskcopy.jl")
include("tarray.jl")

export Trace, current_trace, fork, forkr, randr, TArray, tzeros,
       localcopy, @suppress_err

mutable struct Trace
  task  ::  Task
  vi    ::  VarInfo
  spl   ::  Union{Nothing, Sampler}
  Trace() = (res = new(); res.vi = VarInfo(); res.spl = nothing; res)
end

# NOTE: this function is called by `forkr`
function (::Type{Trace})(f::Function)
  res = Trace();
  # Task(()->f());
  res.task = Task( () -> begin res=f(); put!(Val{:done}); res; end )
  if isa(res.task.storage, Nothing)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

function (::Type{Trace})(f::Function, spl::Sampler, vi :: VarInfo)
  res = Trace();
  res.spl = spl
  # Task(()->f());
  res.vi = deepcopy(vi)
  res.vi.num_produce = 0
  res.task = Task( () -> begin vi_new=f(vi, spl); put!(Val{:done}); vi_new; end )
  if isa(res.task.storage, Nothing)
    res.task.storage = IdDict() # what is the parametrization of the IdDict?
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

# step to the next observe statement, return log likelihood
Base.take!(t::Trace) = (t.vi.num_produce += 1; Base.take!(t.task))

# Task copying version of fork for Trace.
function fork(trace :: Trace, is_ref :: Bool = false)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  if is_ref
    set_retained_vns_del_by_spl!(newtrace.vi, newtrace.spl)
  end

  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace :: Trace)
  newtrace = Trace(trace.task.code)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  newtrace.vi.num_produce = 0

  newtrace
end

current_trace() = current_task().storage[:turing_trace]

end
