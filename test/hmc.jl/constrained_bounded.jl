using Turing
using Test

obs = [0,1,0,1,1,1,1,1,1,1]

@model constrained_test(obs) = begin
  p ~ Beta(2,2)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
  p
end

chain = sample(constrained_test(obs), HMC(1000, 1.5, 3)) # using a large step size (1.5)

#check_numerical(chain, [:p], [10/14], eps=0.1)
