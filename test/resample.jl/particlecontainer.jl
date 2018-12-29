# Test ParticleContainer

using Turing

using Turing.Core.Container: ParticleContainer, weights, resample!, effectiveSampleSize, Trace, current_trace
using Turing.Core.VarReplay: VarName
using Turing.Samplers: Sampler
using Libtask: consume, produce
using Turing.Inference: assume

if isdefined((@static VERSION < v"0.7.0-DEV.484" ? current_module() : @__MODULE__), :n)
  n[] = 0
else
  const n = Ref(0)
end

alg = PG(5, 1)
spl = Sampler(alg)
dist = Normal(0, 1)

function fpc()
  t = TArray(Float64, 1);
  t[1] = 0;
  while true
    ct = current_trace()
    vn = VarName(gensym(), :x, "[$n]", 1)
    assume(spl, dist, vn, ct.vi); n[] += 1;
    produce(0)
    vn = VarName(gensym(), :x, "[$n]", 1)
    assume(spl, dist, vn, ct.vi); n[] += 1;
    t[1] = 1 + t[1]
  end
end

pc = ParticleContainer{Trace}(fpc)

push!(pc, Trace(pc.model))
push!(pc, Trace(pc.model))
push!(pc, Trace(pc.model))

Base.@assert weights(pc)[1] == [1/3, 1/3, 1/3]
Base.@assert weights(pc)[2] ≈ log(3)
Base.@assert pc.logE ≈ log(1)

Base.@assert consume(pc) == log(1)

resample!(pc)
Base.@assert pc.num_particles == length(pc)
Base.@assert weights(pc)[1] == [1/3, 1/3, 1/3]
Base.@assert weights(pc)[2] ≈ log(3)
Base.@assert pc.logE ≈ log(1)
Base.@assert effectiveSampleSize(pc) == 3

Base.@assert consume(pc) ≈ log(1)
resample!(pc)
Base.@assert consume(pc) ≈ log(1)
