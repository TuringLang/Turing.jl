using Distributions
using Turing
using Base.Test

obs = [0,1,0,1,1,1,1,1,1,1]

@model constrained_test begin
  @assume p ~ Beta(2,2)
  @assume x ~ Bernoulli(p)
  for i = 1:length(obs)
    @observe obs[i] ~ Bernoulli(p)
  end
  @predict p x
end

chain = sample(constrained_test, HMC(100, 1.5, 1))  # using a large step size (1.5)
@test_approx_eq_eps mean(res[:p]) 0.716 0.05          # 0.716 is from SMC(10000)
