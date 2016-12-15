using Distributions
using Turing
using Base.Test

@model hmcmatrixsup begin
  @assume p ~ Wishart(7, [1 0.5; 0.5 1])
  @predict p
end

ps = []
for _ in 1:5
  chain = sample(hmcmatrixsup, HMC(3000, 0.3, 3))
  push!(ps, mean(chain[:p]))
end

@test_approx_eq_eps mean(ps) (7 * [1 0.5; 0.5 1]) 0.5
