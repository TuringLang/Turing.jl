using Distributions
using Turing
using Base.Test

obs = [1,2,1,2,2,2,2,2,2,2]

@model constrained_simplex_test begin
  @assume ps ~ Dirichlet(2, 3)
  for i = 1:length(obs)
    @observe obs[i] ~ Categorical(ps)
  end
  @predict ps
end

chain = sample(constrained_simplex_test, HMC(3000, 0.01, 3))  # using a large step size (1.5)
mean(chain[:ps])
