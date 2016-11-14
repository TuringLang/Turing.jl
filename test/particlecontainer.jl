# Test ParticleContainer

using Turing
using Distributions

import Turing: ParticleContainer, weights, resample!, effectiveSampleSize, TraceC, TraceR, Trace, current_trace

function f()
  t = TArray(Float64, 1);
  t[1] = 0;
  while true
    rand(current_trace(), Normal(0,1))
    produce(0)
    rand(current_trace(), Normal(0,1))
    t[1] = 1 + t[1]
  end
end

pc = ParticleContainer{TraceC}(f)

push!(pc, TraceC(pc.model))
push!(pc, TraceC(pc.model))
push!(pc, TraceC(pc.model))

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
