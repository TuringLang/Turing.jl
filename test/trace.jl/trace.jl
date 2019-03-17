# Test Trace

using Turing

import Turing: Trace, current_trace, fork, VarName, Sampler

if isdefined((@static VERSION < v"0.7.0-DEV.484" ? current_module() : @__MODULE__), :n)
  n[] = 0
else
  const n = Ref(0)
end

alg = PG(5, 1)
spl = Turing.Sampler(alg, Turing.VarInfo())
dist = Normal(0, 1)

function f2()
  t = TArray(Int, 1);
  t[1] = 0;
  while true
    ct = current_trace()
    vn = VarName(gensym(), :x, "[$n]", 1)
    Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
    produce(t[1]);
    vn = VarName(gensym(), :x, "[$n]", 1)
    Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
    t[1] = 1 + t[1]
  end
end

# Test task copy version of trace
t = Trace(f2, Turing.VarInfo())

consume(t); consume(t)
a = fork(t);
consume(a); consume(a)

Base.@assert consume(t) == 2
Base.@assert consume(a) == 4
