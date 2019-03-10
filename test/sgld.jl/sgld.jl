using Test, Random, Distributions, Turing

Random.seed!(125)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

chain = sample(gdemo([1.5, 2.0]), SGLD(10000, 0.5))

# Note: samples are weigthed by step sizes cf 4.2 in paper
v = get(chain, [:lf_eps, :s, :m])
s_res1weightedMean = sum(v.lf_eps .* v.s) / sum(v.lf_eps)
m_res1weightedMean = sum(v.lf_eps .* v.m) / sum(v.lf_eps)
@test s_res1weightedMean ≈ 49/24 atol=0.2
@test m_res1weightedMean ≈ 7/6 atol=0.2
