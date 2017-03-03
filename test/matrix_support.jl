using Distributions
using Turing
using Base.Test

@model hmcmatrixsup begin
  v ~ Wishart(7, [1 0.5; 0.5 1])
  v
end

vs = []
for _ in 1:5
  chain = sample(hmcmatrixsup, HMC(1000, 0.3, 2))
  push!(vs, mean(chain[:v]))
end

@test_approx_eq_eps mean(vs) (7 * [1 0.5; 0.5 1]) 0.5
