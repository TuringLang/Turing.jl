using Distributions
using Turing
using Base.Test

data = Dict(:obs=>[0, 1, 0, 1, 1, 1, 1, 1, 1, 1])

@model newinterface begin
  p ~ Beta(2,2)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
end
Turing.TURING[:modelex]
ga = GradientInfo()
sampler = HMCSampler{HMC}(HMC(100, 1.5, 3))
ga = newinterface(ga, data, sampler)
newinterface

chain = sample(newinterface, HMC(100, 1.5, 3))

chain = sample(
  constrained_test,
  HMC(3000, 1.5, 3; :p)
)

 # using a large step size (1.5)
@test_approx_eq_eps mean(chain[:p]) 10/14 0.10
