using Distributions
using Turing
using Base.Test

@model hmcmatrixsup begin
  @assume p ~ Wishart(7, [1 0.5; 0.5 1])
  @predict p
end

chain = sample(hmcmatrixsup, HMC(3000, 0.3, 3))
@test_approx_eq_eps mean(chain[:p]) (7 * [1 0.5; 0.5 1]) 0.1
