"""
Notes:
 - `rand` will store randomness only when trace type matches TraceR.
 - `randc` never stores randomness. [REMOVED]
 - `randr` will store and replay randomness regardless trace type (N.B. Particle Gibbs uses `randr`).
 - `fork1` will perform replaying immediately and fix the particle weight to 1.
"""

module Traces
using Turing: VarInfo, Sampler, getvns, NULL, getretain

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

export Trace, TraceR, TraceC, current_trace, fork, randr, TArray, tzeros,
       localcopy, @suppress_err

type Trace{T}
  task  ::  Task
  vi    ::  VarInfo
  spl   ::  Union{Void, Sampler}
  Trace{T}() where {T} = (res = new(); res.vi = VarInfo(); res.spl = nothing; res)
end

# NOTE: this function is called by `forkr`
function (::Type{Trace{T}}){T}(f::Function)
  res = Trace{T}();
  # Task(()->f());
  res.task = Task( () -> begin res=f(); produce(Val{:done}); res; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

function (::Type{Trace{T}}){T}(f::Function, spl::Sampler, vi :: VarInfo)
  res = Trace{T}();
  res.spl = spl
  # Task(()->f());
  res.vi = deepcopy(vi)
  res.vi.num_produce = 0
  res.task = Task( () -> begin vi_new=f(vi, spl); produce(Val{:done}); vi_new; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

const TraceR = Trace{:R} # Task Copy
const TraceC = Trace{:C} # Replay

# step to the next observe statement, return log likelihood
Base.consume(t::Trace) = (t.vi.num_produce += 1; Base.consume(t.task))

# Task copying version of fork for both TraceR and TraceC.
function forkc(trace :: Trace, is_ref :: Bool = false)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  if is_ref
    newtrace.vi[getretain(newtrace.vi, newtrace.spl)] = NULL
  end

  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# PG requires for the reference particle keeping all randomness
function forkr(trace :: TraceR)
  # Create new task and copy randomness
  newtrace = TraceR(trace.task.code)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  newtrace.vi.num_produce = 0

  newtrace
end

# Default fork implementation.
fork(s :: TraceR) = forkr(s)
fork(s :: TraceC) = forkc(s)

# Note that:
#  - lazy replay is only possible for TraceR
#  - lazy replay is useful for implementing PG (i.e. ref particle)

current_trace() = current_task().storage[:turing_trace]

end
