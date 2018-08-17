using Turing
using Test
using Pkg

include(Pkg.dir("Turing")*"/deps/deps.jl")
check_deps()
libpath = dirname(libtask)

if !(libpath in Base.DL_LOAD_PATH)
  push!(Base.DL_LOAD_PATH, libpath)
end

function produce(v)
  tsk = current_task()

  @assert isa(tsk.storage, IdDict)
  @assert haskey(tsk.storage, :Channel)

  tsk.storage[:Cond] = Condition()
  put!(tsk.storage[:Channel], (v, string(tsk)) )
  
  wait(tsk.storage[:Cond])
end

function consume(tsk::Task)
 
  @assert isa(tsk.storage, IdDict)
  @assert haskey(tsk.storage, :Channel)

  notify(tsk.storage[:Cond])
  return take!(tsk.storage[:Channel])
end

function f()
  n = 0
  while true
    n += 1 # some computation
    # put the output of the computation into the Channel
    produce(n)
  end
end

function test()

  # some print outs

  # create the Channel for Task communication
  c = Channel(1)

  # create a task for f()
  t = Task(f)

  # add the Channel to the Task's storage
  t.storage = IdDict()
  t.storage[:Channel] = c

  @test !istaskstarted(t)

  # add the Task to the scheduler
  schedule(t)
  yield()

  @test istaskstarted(t)

  # test
  for i in 1:5
    @test consume(t)[1] == i
  end

  @test !istaskdone(t)
  
  # switch to the scheduler to allow another scheduled task to run
  # without this switch, the Task t is queued and not runnable
  yield()

  # copy the task
  t2 = Turing.copy(t)

  schedule(t2)
  yield()

  @test pointer_from_objref(t) != pointer_from_objref(t2)
  @info "successfully copied $t to $(t2)"

	# not sure why the task is not started...
  println(istaskstarted(t2))
  
	@test consume(t)[2] == string(t)
  @test consume(t2)[2] == string(t2)

end

test()

