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

include("taskcopy.jl")
include("tarray.jl")

export Trace, TraceR, TraceC, current_trace, fork, fork2, randr, TArray, tzeros,
       localcopy

type Trace{T}
  task :: Task
  randomness :: Array{Any, 1}    # elem t is the randomness created by the t’th assume call.
  index :: Int64                 # index of current randomness
  num_produce :: Int64           # num of produce calls from trace, each produce corresponds to an observe.
  Trace() = (res = new(); res.randomness = Array{Any,1}(); res.index = 0; res.num_produce = 0; res)
end

function call{T}(::Type{Trace{T}}, f::Function)
  res = Trace{T}();
  res.task = Task(()->f());
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

typealias TraceR Trace{:R} # Task Copy
typealias TraceC Trace{:C} # Replay

# generate a new random variable, replay if t.counter < length(t.randomness)
function randr( t::Trace, distr :: Distribution )
  t.index += 1
  if t.index < length(t.randomness)
    res = t.randomness[t.index]
  else # sample, record
    res = Distributions.rand(distr)
    push!(t.randomness, res)
  end
  return res
end

# generate a new random variable, no replay
randc(t::Trace, distr :: Distribution) = Distributions.rand(distr)

Distributions.rand(t::TraceR, distr :: Distribution) = randr(t, distr)
Distributions.rand(t::TraceC, distr :: Distribution) = randc(t, distr)

# step to the next observe statement, return log likelihood
Base.consume(t::Trace) = (t.num_produce += 1; Base.consume(t.task))

# Task copying version of fork for both TraceR and TraceC.
function forkc(trace :: Trace)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  n_rand = min(trace.index, length(trace.randomness))
  newtrace.randomness = trace.randomness[1:n_rand]
  newtrace.index = trace.index
  newtrace.num_produce = trace.num_produce
  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# fork s and replay until observation t; drop randomness between y_t:T if keep == false
#  N.B.: PG requires keeping all randomness even we only replay up to observation y_t
function forkr(trace :: TraceR, t :: Int64, keep :: Bool)
  # Step 0: create new task and copy randomness
  newtrace = TraceR(trace.task.code)
  newtrace.randomness = deepcopy(trace.randomness)

  # Step 1: Call consume t times to replay randomness
  map((i)->consume(newtrace), 1:t)

  # Step 2: Remove remaining randomness if keep==false
  if !keep
    newtrace.randomness = newtrace.randomness[1:newtrace.index]
  end
  newtrace
end

# Default fork implementation, replay immediately.
fork(s :: TraceR) = forkr(s, s.num_produce, false)
fork(s :: TraceC) = forkc(s)

# Lazily replay on demand, note that:
#  - lazy replay is only possible for TraceR
#  - lazy replay accumulates likelihoods
#  - lazy replay is useful for implementing PG (i.e. ref particle)
fork2(s :: TraceR) = forkr(s, 0, true)

current_trace() = current_task().storage[:turing_trace]

end
