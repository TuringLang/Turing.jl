using Turing, DataFrames, Gadfly, Distributions

# HMM example from Anglican and Probabilistic C papers

statesmean = [-1, 1, 0]
initial    = Categorical([1.0/3, 1.0/3, 1.0/3])
trans      = [Categorical([0.1, 0.5, 0.4]), Categorical([0.2, 0.2, 0.6]), Categorical([0.15, 0.15, 0.7])]
data       = [0, 0.9, 0.8, 0.7, 0, -0.025, -5, -2, -0.1, 0, 0.13]

@model hmmdemo begin
  states = TArray(Int, length(data))
  @assume states[1] ~ initial
  for i = 2:length(data)
    @assume states[i] ~ trans[states[i-1]]
    @observe data[i]  ~ Normal(statesmean[states[i]], 0.4)
  end
  @predict states data
end

#  run sampler, collect results
@time chain1  = sample(hmmdemo, SMC(500))
@time chain2  = sample(hmmdemo, PG(10, 20))

SMCInferedStates  = mean([statesmean[v] for v in chain1[:states]])
PGInferedStates   = mean([statesmean[v] for v in chain2[:states]])
println("Infered states  (smc)", round(SMCInferedStates, 2))
println("Infered states   (pg)", round(PGInferedStates, 2))
println("True states         ", round(data+0.01, 2))
