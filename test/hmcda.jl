using Turing
using Base.Test

alg1 = HMCDA(3000, 200, 0.65, 0.2)
# alg2 = Gibbs(3000, HMCDA(1, 200, 0.65, 0.35, :m), HMC(1, 0.25, 3, :s))
# alg3 = Gibbs(3000, PG(50, 10, :s), HMCDA(1, 500, 0.65, 0.05, :m))
# alg3 = Gibbs(2000, HMC(1, 0.25, 3, :m), PG(30, 3, :s))
# alg3 = PG(50, 2000)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

res1 = sample(gdemo([1.5, 2.0]), alg1)
println("E[s] = $(mean(res1[:s]))")
println("E[m] = $(mean(res1[:m]))")
@test_approx_eq_eps mean(res1[:s]) 49/24 0.2
@test_approx_eq_eps mean(res1[:m]) 7/6 0.2

# res2 = sample(gdemo([1.5, 2.0]), alg2)
#
# @test_approx_eq_eps mean(res2[:s]) 49/24 0.2
# @test_approx_eq_eps mean(res2[:m]) 7/6 0.2

# res3 = sample(gdemo([1.5, 2.0]), alg3)
#
# @test_approx_eq_eps mean(res3[:m]) 7/6 0.15
# @test_approx_eq_eps mean(res3[:s]) 49/24 0.15

res1 = sample(gdemo([1.5, 2.0]), HMC(3000, 0.2, 4))
println("HMC")
println("E[s] = $(mean(res1[:s]))")
println("E[m] = $(mean(res1[:m]))")
