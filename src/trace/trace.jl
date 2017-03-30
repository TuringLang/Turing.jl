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
using Turing: VarName, VarInfo
import Turing.randr

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
  task :: Task
  vi   :: VarInfo
  Trace() = (res = new(); res.vi = VarInfo(); res)
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

function (::Type{Trace{T}}){T}(f::Function, data, spl, vi :: VarInfo)
  res = Trace{T}();
  # Task(()->f());
  res.vi.idcs = vi.idcs
  res.vi.vals = vi.vals
  res.vi.syms = vi.syms
  res.vi.dists = vi.dists
  res.task = Task( () -> begin res=f(data, vi, spl); produce(Val{:done}); res; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

typealias TraceR Trace{:R} # Task Copy
typealias TraceC Trace{:C} # Replay

# generate a new random variable, replay if t.counter < length(t.randomness)

randr(t::Trace, vn::VarName, distr::Distribution) = randr(t.vi, vn, distr)

# generate a new random variable, no replay
randc(t::Trace, distr :: Distribution) = Distributions.rand(distr)

Distributions.rand(t::TraceR, vn::VarName, dist::Distribution) = randr(t, vn, dist)
Distributions.rand(t::TraceC, vn::VarName, dist::Distribution) = randc(t, dist)

Distributions.rand(t::TraceR, distr :: Distribution) = randr(t, distr)
Distributions.rand(t::TraceC, distr :: Distribution) = randc(t, distr)

# step to the next observe statement, return log likelihood
Base.consume(t::Trace) = (t.vi.num_produce += 1; Base.consume(t.task))

# Task copying version of fork for both TraceR and TraceC.
function forkc(trace :: Trace)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  n_rand = min(trace.vi.index, length(trace.vi.randomness))
  newtrace.vi.idcs = trace.vi.idcs
  newtrace.vi.vals = trace.vi.vals
  newtrace.vi.syms = trace.vi.syms
  newtrace.vi.dists = trace.vi.dists
  newtrace.vi.randomness = trace.vi.randomness[1:n_rand]
  newtrace.vi.names = trace.vi.names[1:n_rand]
  newtrace.vi.tsyms = trace.vi.tsyms[1:n_rand]
  newtrace.vi.index = trace.vi.index
  newtrace.vi.num_produce = trace.vi.num_produce
  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# fork s and replay until observation t; drop randomness between y_t:T if keep == false
#  N.B.: PG requires keeping all randomness even we only replay up to observation y_t
function forkr(trace :: TraceR, t :: Int, keep :: Bool)
  # Step 0: create new task and copy randomness
  newtrace = TraceR(trace.task.code)
  newtrace.vi.idcs = trace.vi.idcs
  newtrace.vi.vals = trace.vi.vals
  newtrace.vi.syms = trace.vi.syms
  newtrace.vi.dists = trace.vi.dists
  newtrace.vi.randomness = deepcopy(trace.vi.randomness)
  # NOTE: use deepcopy below directly will cause error
  newtrace.vi.names = map(vn -> identity(vn), trace.vi.names)

  newtrace.vi.tsyms = deepcopy(trace.vi.tsyms)

  # Step 1: Call consume t times to replay randomness
  map(i -> consume(newtrace), 1:t)

  # Step 2: Remove remaining randomness if keep==false
  if !keep
    index = newtrace.vi.index
    newtrace.vi.randomness = newtrace.vi.randomness[1:index]
    newtrace.vi.names = newtrace.vi.names[1:index]
    newtrace.vi.tsyms = newtrace.vi.tsyms[1:index]
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
