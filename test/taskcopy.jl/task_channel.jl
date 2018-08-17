using Turing

function f()

  n = 0

  tsk = current_task()
  @assert isa(tsk.storage, IdDict)
  @assert haskey(tsk.storage, :Channel)
  
  while true
    n += 1 # some computation

    put!(tsk.storage[:Channel], n)
  end
end

function test()

  c = Channel(1)
  t = Task(f)
  t.storage = IdDict()
  t.storage[:Channel] = c

  yield(t)

  println(take!(c))
  println(take!(c))
  println(take!(c))
end

