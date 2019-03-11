using Test, Random, Distributions, Turing

Random.seed!(125)

alg = SGHMC(10000, 0.02, 0.5)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

chain = sample(gdemo([1.5, 2.0]), alg)

@test mean(chain[:s].value) ≈ 49/24 atol=0.1
@test mean(chain[:m].value) ≈ 7/6 atol=0.1
