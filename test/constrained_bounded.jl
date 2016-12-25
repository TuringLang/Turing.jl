using Distributions
using Turing
using Base.Test

obs = [0,1,0,1,1,1,1,1,1,1]

@model constrained_test begin
  p ~ Beta(2,2)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
  @predict p
end

chain = sample(constrained_test, HMC(3000, 1.5, 3)) # using a large step size (1.5)
@test_approx_eq_eps mean(chain[:p]) 10/14 0.10      # 0.716 is from SMC(10000)
