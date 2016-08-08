@unix_only libpath = replace(@__FILE__, "src/trace/taskcopy.jl", "deps/")
@windows_only libpath = replace(@__FILE__, "src\\trace\\taskcopy.jl", "deps\\")

if !(ByteString(libpath) in Base.DL_LOAD_PATH)
  push!(Base.DL_LOAD_PATH, ByteString(libpath))
end
# println(libpath)

function sweepandmark(t::Task)
  s :: ObjectIdDict = t.storage
  for k in keys(s)
    # All copy-on-write family data structures stores data in task_local_storage in
    # the form of (taskid::Task, data::Any). Setting taskid to nothing will triger
    #  copy-on-write for the parent task (which is copied).
    if isa(s[k], Tuple{Union{Void,Task}, Any})
      _, d = s[k]
      s[k] = (nothing, d)
    end
  end
end

function Base.copy(t::Task)
  t.state != :runnable && t.state != :done &&
    error("Only runnable or finished tasks can be copied.")
  newt = ccall((:jl_clone_task, "libtask"), Any, (Any,), t)::Task
  if t.storage != nothing
    newt.storage = copy(t.storage)
    sweepandmark(t)
  else
    newt.storage = nothing
  end
  newt.code = t.code
  newt.state = t.state
  newt.result = t.result
  newt.parent = t.parent
  newt
end

olderr = STDERR
# supress warning message for re-defining Base.produce
redirect_stderr()
function Base.produce(v)
  #### un-optimized version
  #q = current_task().consumers
  #t = shift!(q.waitq)
  #empty = isempty(q.waitq)
  ct = current_task()
  local empty, t, q
  while true
    q = ct.consumers
    if isa(q,Task)
      t = q
      ct.consumers = nothing
      empty = true
      break
    elseif isa(q,Condition) && !isempty(q.waitq)
      t = shift!(q.waitq)
      empty = isempty(q.waitq)
      break
    end
    wait()
  end

  t.state = :runnable
  if empty
    if isempty(Base.Workqueue)
      yieldto(t, v)
    else
      Base.schedule_and_wait(t, v)
    end
    ct = current_task() # When a task is copied, ct should be updated to new task ID.
    while true
      # wait until there are more consumers
      q = ct.consumers
      if isa(q,Task)
        return q.result
      elseif isa(q,Condition) && !isempty(q.waitq)
        return q.waitq[1].result
      end
      wait()
    end
  else
    schedule(t, v)
    # make sure `t` runs before us. otherwise, the producer might
    # finish before `t` runs again, causing it to see the producer
    # as done, causing done(::Task, _) to miss the value `v`.
    # see issue #7727
    yield()
    return q.waitq[1].result
  end
end
redirect_stderr(olderr)

