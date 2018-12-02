using Turing.VarReplay

mutable struct Trace
  task  ::  Task
  vi    ::  VarInfo
  spl   ::  Union{Nothing, Sampler}
  Trace() = (res = new(); res.vi = VarInfo(); res.spl = nothing; res)
end

# NOTE: this function is called by `forkr`
function Trace(f)
  res = Trace();
  # Task(()->f());
  res.task = Task( () -> begin res=f(); produce(Val{:done}); res; end )
  if isa(res.task.storage, Nothing)
    res.task.storage = IdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

function Trace(f, spl::Sampler, vi :: VarInfo)
  res = Trace();
  res.spl = spl
  # Task(()->f());
  res.vi = deepcopy(vi)
  res.vi.num_produce = 0
  res.task = Task( () -> begin vi_new=f(vi, spl); produce(Val{:done}); vi_new; end )
  if isa(res.task.storage, Nothing)
    res.task.storage = IdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

# step to the next observe statement, return log likelihood
Turing.consume(t::Trace) = (t.vi.num_produce += 1; consume(t.task))

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
