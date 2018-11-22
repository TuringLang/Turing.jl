include("../utility.jl")
using Test
using Turing

@model gdemo(x) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

model_f = gdemo([1.5, 2.0])

alg = NUTS(5000, 1000, 0.65)
res = sample(model_f, alg)

@info(mean(res[:s][1000:end])," ≈ ", 49/24, "?")
@info(mean(res[:m][1000:end])," ≈ ", 7/6, "?")
@test mean(res[:s][1000:end]) ≈ 49/24 atol=0.2
@test mean(res[:m][1000:end]) ≈ 7/6 atol=0.2
