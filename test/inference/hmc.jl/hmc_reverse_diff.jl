using Turing
using Test

@model gdemo(x) = begin
  s ~ InverseGamma(2, 3)
  # println(s)
  m ~ Normal(0, sqrt(s))
  # println(m)
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  # println(vi)
  return s, m
end

alg = HMC(3000, 0.15, 10)

res = sample(gdemo_default, alg)

# println(mean(res[:s].value)," ≈ ", 49/24, "?")
# println(mean(res[:m].value)," ≈ ", 7/6, "?")
# @test mean(res[:s].value) ≈ 49/24 atol=0.2
# @test mean(res[:m].value) ≈ 7/6 atol=0.2
check_numerical(chain, [:s, :m], [49/24, 7/6], eps=0.1)
