using Turing
using Test

obs12 = [1,2,1,2,2,2,2,2,2,2]

@model constrained_simplex_test(obs12) = begin
  ps ~ Dirichlet(2, 3)
  for i = 1:length(obs12)
    obs12[i] ~ Categorical(ps)
  end
  return ps
end

chain = sample(constrained_simplex_test(obs12), HMC(1000, 0.75, 2))

#check_numerical(chain, [:ps], [[5/16; 11/16]], eps=0.015)
