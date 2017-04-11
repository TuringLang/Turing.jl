"""
Notes:
 - `rand` will store randomness only when trace type matches TraceR.
 - `randc` never stores randomness.
 - `randr` will store and replay randomness regardless trace type (N.B. Particle Gibbs uses `randr`).
 - `fork1` will perform replaying immediately and fix the particle weight to 1.
 - `fork2` will perform lazy replaying and accumulate likelihoods like a normal particle.
"""

module Traces
using Distributions
using Turing: VarName, VarInfo, Sampler, retain, groupvals
import Turing.randr, Turing.randoc

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

export Trace, TraceR, TraceC, current_trace, fork, fork2, randr, TArray, tzeros,
       localcopy, @suppress_err

type Trace{T}
  task  ::  Task
  vi    ::  VarInfo
  spl   ::  Union{Void, Sampler}
  Trace() = (res = new(); res.vi = VarInfo(); res.spl = nothing; res)
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
  res.vi.index = 0
  res.vi.num_produce = 0
  res.task = Task( () -> begin res=f(vi=vi, sampler=spl); produce(Val{:done}); res; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

typealias TraceR Trace{:R} # Task Copy
typealias TraceC Trace{:C} # Replay

# generate a new random variable, replay if t.counter < length(t.randomness)
randr(t::Trace, vn::VarName, distr::Distribution) = randr(t.vi, vn, distr, 0, nothing, true)

# generate a new random variable, no replay
randc(t::Trace, vn::VarName, distr :: Distribution) = randoc(t.vi, vn, distr)

Distributions.rand(t::TraceR, vn::VarName, dist::Distribution) = randr(t, vn, dist)
Distributions.rand(t::TraceC, vn::VarName, dist::Distribution) = randc(t, vn, dist)

Distributions.rand(t::TraceR, distr :: Distribution) = randr(t, distr)
Distributions.rand(t::TraceC, distr :: Distribution) = randc(t, distr)

# step to the next observe statement, return log likelihood
Base.consume(t::Trace) = (t.vi.num_produce += 1; Base.consume(t.task))

# Task copying version of fork for both TraceR and TraceC.
function forkc(trace :: Trace)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  newtrace.spl = trace.spl
  if trace.spl != nothing
    gid = trace.spl.alg.group_id
  else
    gid = 0
  end

  n_rand = min(trace.vi.index, length(groupvals(trace.vi, gid, trace.spl)))
  newtrace.vi = retain(deepcopy(trace.vi), gid, n_rand)
  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# fork s and replay until observation t; drop randomness between y_t:T if keep == false
#  N.B.: PG requires keeping all randomness even we only replay up to observation y_t
function forkr(trace :: TraceR, t :: Int, keep :: Bool)
  # Step 0: create new task and copy randomness
  newtrace = TraceR(trace.task.code)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  newtrace.vi.index = 0
  newtrace.vi.num_produce = 0

  # Step 1: Call consume t times to replay randomness
  map(i -> consume(newtrace), 1:t)

  # Step 2: Remove remaining randomness if keep==false
  if !keep
    index = newtrace.vi.index
    if trace.spl != nothing
      gid = trace.spl.alg.group_id
    else
      gid = 0
    end
    retain(newtrace.vi, gid, index)
  end

  newtrace
end

# Default fork implementation, replay immediately.
fork(s :: TraceR) = forkr(s, s.vi.num_produce, false)
fork(s :: TraceC) = forkc(s)

# Lazily replay on demand, note that:
#  - lazy replay is only possible for TraceR
#  - lazy replay accumulates likelihoods
#  - lazy replay is useful for implementing PG (i.e. ref particle)
fork2(s :: TraceR) = forkr(s, 0, true)

current_trace() = current_task().storage[:turing_trace]

end
