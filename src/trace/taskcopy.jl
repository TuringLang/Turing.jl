using Turing: libtask

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
  newt = ccall((:jl_clone_task, libtask), Any, (Any,), t)::Task
  if t.storage != nothing
    n = n_copies(t)
    t.storage[:n_copies]  = 1 + n
    newt.storage = copy(t.storage)
  else
    newt.storage = nothing
  end
  # copy fields not accessible in task.c
  newt.code = t.code
  newt.state = t.state
  newt.result = t.result
  newt.parent = t.parent
  if :last in fieldnames(typeof(t))
    newt.last = nothing
  end
  newt
end

produce(v) = begin
    ct = current_task()

    if ct.storage == nothing
        ct.storage = IdDict()
    end

    haskey(ct.storage, :consumers) || (ct.storage[:consumers] = nothing)
    local empty, t, q
    while true
        q = ct.storage[:consumers]
        if isa(q,Task)
            t = q
            ct.storage[:consumers] = nothing
            empty = true
            break
        elseif isa(q,Condition) && !isempty(q.waitq)
            t = shift!(q.waitq)
            empty = isempty(q.waitq)
            break
        end
        wait()
    end

    t.state == :runnable || throw(AssertionError("producer.consumer.state == :runnable"))
    if empty
        Base.schedule_and_wait(t, v)
        ct = current_task() # When a task is copied, ct should be updated to new task ID.
        while true
            # wait until there are more consumers
            q = ct.storage[:consumers]
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

consume(p::Task, values...) = begin

    if p.storage == nothing
        p.storage = IdDict()
    end
    haskey(p.storage, :consumers) || (p.storage[:consumers] = nothing)

    if istaskdone(p)
        return wait(p)
    end

    ct = current_task()
    ct.result = length(values)==1 ? values[1] : values

    #### un-optimized version
    #if P.consumers === nothing
    #    P.consumers = Condition()
    #end
    #push!(P.consumers.waitq, ct)
    # optimized version that avoids the queue for 1 consumer
    consumers = p.storage[:consumers]
    if consumers === nothing || (isa(consumers,Condition)&&isempty(consumers.waitq))
        p.storage[:consumers] = ct
    else
        if isa(consumers, Task)
            t = consumers
            p.storage[:consumers] = Condition()
            push!(p.storage[:consumers].waitq, t)
        end
        push!(p.storage[:consumers].waitq, ct)
    end

    p.state == :runnable ? Base.schedule_and_wait(p) : wait() # don't attempt to queue it twice
end
