using Turing
using Base.Test

alg1 = HMCDA(2000, 200, 0.65, 0.35)
alg2 = Gibbs(2000, HMCDA(1, 200, 0.65, 0.35, :m), HMC(1, 0.25, 3, :s))
alg3 = Gibbs(2000, HMCDA(1, 200, 0.65, 0.35, :m), PG(30, 3, :s))

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

# res1 = sample(gdemo([1.5, 2.0]), alg1)
#
# @test_approx_eq_eps mean(res1[:s]) 49/24 0.05
# @test_approx_eq_eps mean(res1[:m]) 7/6 0.05
#
# res2 = sample(gdemo([1.5, 2.0]), alg2)
#
# @test_approx_eq_eps mean(res2[:s]) 49/24 0.05
# @test_approx_eq_eps mean(res2[:m]) 7/6 0.05

res3 = sample(gdemo([1.5, 2.0]), alg3)

@test_approx_eq_eps mean(res3[:s]) 49/24 0.05
@test_approx_eq_eps mean(res3[:m]) 7/6 0.05
