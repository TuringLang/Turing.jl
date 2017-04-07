# Test TraceC and TraceR

using Turing
using Distributions

import Turing: Trace, TraceR, TraceC, current_trace, fork, fork2, randr, addvar!, VarName

global n = 0



function f2()
  global n
  t = TArray(Int, 1);
  t[1] = 0;
  while true
    ct = current_trace()
    vn = VarName(gensym(), :x, "[$n]", 1)
    rand(ct, vn, Normal(0,1)); n += 1;
    produce(t[1]);
    vn = VarName(gensym(), :x, "[$n]", 1)
    rand(ct, vn, Normal(0,1)); n += 1;
    t[1] = 1 + t[1]
  end
end

# Test task copy version of trace
t = TraceC(f2)

consume(t); consume(t)
a = fork(t);
consume(a); consume(a)

Base.@assert consume(t) == 2
Base.@assert consume(a) == 4


# Test replaying version of trace
t = TraceR(f2)

consume(t); consume(t)
a = fork(t);
consume(a); consume(a)

Base.@assert consume(t) == 2
Base.@assert consume(a) == 4

a2 = fork(t)
Base.@assert length(a2.vi.vals) == 5
Base.@assert t.vi.vals == a2.vi.vals
Base.@assert t.vi.index == a2.vi.index
