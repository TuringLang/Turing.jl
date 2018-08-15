# Test task copying

using  Turing
import Turing.Traces: produce,consume

# Test case 1: stack allocated objects are deep copied.
function f_ct()
  t::Int64 = 0;
  while true
    produce(t)
    t = 1 + t
  end
end

t = Task(f_ct)

consume(t); consume(t)
a = copy(t);


schedule(a)

consume(a); consume(a)

# Test case 2: heap allocated objects are shallowly copied.

function f_ct2()
  t = [0 1 2];
  while true
    #println(pointer_from_objref(t)); REVIEW: can we remove this comments (Kai)
    produce(t[1])
    t[1] = 1 + t[1]
  end
end

t = Task(f_ct2)

consume(t); consume(t)
a = copy(t);
consume(a); consume(a)

# REVIEW: comments below need to be updated (Kai)
# more: add code in copy() to handle invalid cases for cloning tasks.

function f_ct3()
  t = [0];
  o = (x) -> x + 1;  # not heap allocated?
  while true
    produce(t[1])
    t[1] = 1 + t[1]
  end
  return o
end

t = Task(f_ct3)

consume(t); consume(t);
a = copy(t);
consume(a); consume(a)
