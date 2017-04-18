using Turing
using Base.Test

alg1 = HMCDA(6000, 200, 0.65, 0.45)
alg2 = HMC(6000, 0.25, 3)
alg3 = Gibbs(6000, HMC(1, 0.25, 3, :s), HMCDA(1, 200, 0.65, 0.45, :m))
alg3 = Gibbs(6000, PG(20, 1, :s), HMCDA(1, 200, 0.65, 0.45, :m))

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

res2 = sample(gdemo([1.5, 2.0]), alg2)

res1 = sample(gdemo([1.5, 2.0]), alg1)

@test_approx_eq_eps mean(res1[:s]) mean(res2[:s]) 0.15
@test_approx_eq_eps mean(res1[:m]) mean(res2[:m]) 0.15

res3 = sample(gdemo([1.5, 2.0]), alg3)

@test_approx_eq_eps mean(res3[:s]) mean(res2[:s]) 0.15
@test_approx_eq_eps mean(res3[:m]) mean(res2[:m]) 0.15

res4 = sample(gdemo([1.5, 2.0]), alg4)

@test_approx_eq_eps mean(res4[:s]) mean(res2[:s]) 0.15
@test_approx_eq_eps mean(res4[:m]) mean(res2[:m]) 0.15
