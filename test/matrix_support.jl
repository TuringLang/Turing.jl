using Distributions
using Turing
using Base.Test

@model hmcmatrixsup begin
  @assume p ~ Wishart(7, [1 0.5; 0.5 1])
  @predict p
end

chain = sample(hmcmatrixsup, HMC(5000, 0.1, 3))

@test_approx_eq_eps mean(chain[:p]) (7 * [1 0.5; 0.5 1]) 0.5
