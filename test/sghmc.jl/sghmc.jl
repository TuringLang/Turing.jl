using Turing
using Base.Test

srand(125)

alg1 = SGHMC(3000, 0.01, 0.5)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  x[1] ~ Normal(m, sqrt.(s))
  x[2] ~ Normal(m, sqrt.(s))
  return s, m
end

res1 = sample(gdemo([1.5, 2.0]), alg1)
println("E[s] = $(mean(res1[:s]))")
println("E[m] = $(mean(res1[:m]))")
@test_approx_eq_eps mean(res1[:s]) 49/24 0.2
@test_approx_eq_eps mean(res1[:m]) 7/6 0.2

res1 = sample(gdemo([1.5, 2.0]), HMC(3000, 0.2, 4))
println("HMC")
println("E[s] = $(mean(res1[:s]))")
println("E[m] = $(mean(res1[:m]))")
