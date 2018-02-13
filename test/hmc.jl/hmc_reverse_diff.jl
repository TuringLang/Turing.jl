using Turing

setadbackend(:reverse_diff)

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s[1]))
  x[1] ~ Normal(m[1], sqrt(s[1]))
  x[2] ~ Normal(m[1], sqrt(s[1]))
  return s, m
end

alg = HMC(30000, 0.005, 10)

res = sample(gdemo([1.5, 2.0]), alg)

println(mean(res[:s]), 49/24)
println(mean(res[:m]), 7/6)
