using Turing

# test case 1: stack allocated objects are deep copied.
function f()
  t = 0;
  while true
    produce(t)
    t = 1 + t
  end
end

t = Task(f)

consume(t); consume(t)
a = copy(t);
consume(a); consume(a)



# test case 2: heap allocated objects are shallowly copied.

function f2()
  t = [0 1 2];
  while true
    #println(pointer_from_objref(t));
    produce(t[1])
    t[1] = 1 + t[1]
  end
end

t = Task(f2)

consume(t); consume(t)
a = copy(t);
consume(a); consume(a)

# more: add code in copy() to handle invalid cases for cloning tasks.

function f3()
  t = [0];
  o = (x) -> x + 1;  # not heap allocated?
  while true
    produce(t[1])
    t[1] = 1 + t[1]
  end
  return o
end

t = Task(f3)

consume(t); consume(t);
a = copy(t);
consume(a); consume(a)
