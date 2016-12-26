using Distributions
using Turing
using Base.Test

data = Dict(:obs=>[0, 1, 0, 1, 1, 1, 1, 1, 1, 1])

@model newinterface begin
  p ~ Beta(2,2)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
  @predict p
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

chain = sample(newinterface, data, HMC(2000, 0.75, 3))

 # using a large step size (1.5)
@test_approx_eq_eps mean(chain[:p]) 10/14 0.10
