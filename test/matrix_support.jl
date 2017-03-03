using Distributions
using Turing
using Base.Test

@model hmcmatrixsup begin
  p ~ Wishart(7, [1 0.5; 0.5 1])
  p
end

p = []
for _ in 1:5
  chain = sample(hmcmatrixsup, HMC(1000, 0.3, 2))
  push!(p, mean(chain[:p]))
end

@test_approx_eq_eps mean(p) (7 * [1 0.5; 0.5 1]) 0.5
