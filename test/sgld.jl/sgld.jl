using Turing
using Base.Test

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

res1 = sample(gdemo([1.5, 2.0]), SGLD(10000, 0.5))
# Note: samples should be weigthed by step sizes cf 4.2 in paper
s_res1weightedMean = sum(res1[:lf_eps] .* res1[:s]) / sum(res1[:lf_eps])
m_res1weightedMean = sum(res1[:lf_eps] .* res1[:m]) / sum(res1[:lf_eps])
println("E[s] = $(s_res1weightedMean)")
println("E[m] = $(m_res1weightedMean)")
@test_approx_eq_eps s_res1weightedMean 49/24 0.2
@test_approx_eq_eps m_res1weightedMean 7/6 0.2

res2 = sample(gdemo([1.5, 2.0]), HMC(3000, 0.2, 4))
println("HMC")
println("E[s] = $(mean(res2[:s]))")
println("E[m] = $(mean(res2[:m]))")
