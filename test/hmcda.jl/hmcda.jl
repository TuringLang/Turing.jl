using Test, Random, Turing

Random.seed!(128)

alg1 = HMCDA(3000, 1000, 0.65, 0.015)
# alg2 = Gibbs(3000, HMCDA(1, 200, 0.65, 0.35, :m), HMC(1, 0.25, 3, :s))
alg3 = Gibbs(1500, PG(30, 10, :s), HMCDA(1, 500, 0.65, 0.005, :m))
# alg3 = Gibbs(2000, HMC(1, 0.25, 3, :m), PG(30, 3, :s))
# alg3 = PG(50, 2000)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

res1 = sample(gdemo([1.5, 2.0]), alg1)

#check_numerical(res1, [:s, :m], [49/24, 7/6])

# res2 = sample(gdemo([1.5, 2.0]), alg2)
#
# @test mean(res2[:s]) ≈ 49/24 atol=0.2
# @test mean(res2[:m]) ≈ 7/6 atol=0.2

res3 = sample(gdemo([1.5, 2.0]), alg3)

#check_numerical(res3, [:s, :m], [49/24, 7/6])

# res1 = sample(gdemo([1.5, 2.0]), HMC(3000, 0.2, 4))
# println("HMC")
# println("E[s] = $(mean(res1[:s]))")
# println("E[m] = $(mean(res1[:m]))")
