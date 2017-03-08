using Distributions
using Turing
using Base.Test

obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

@model newinterface(obs) begin
  p ~ Beta(2,2)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
  p
end

Turing.TURING[:modelex]

# newinterface(data)
#
# ga = VarInfo()
# sampler = HMCSampler{HMC}(HMC(100, 1.5, 3))
# ga = newinterface(data, ga, sampler)
# newinterface
#
# chain = sample(newinterface, HMC(100, 1.5, 3))

chain = @sample(newinterface(obs), HMC(100, 0.75, 3, :p, :x))
