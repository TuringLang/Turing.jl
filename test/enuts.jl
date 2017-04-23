using Distributions, Turing

@model gdemo(x) = begin
  # s ~ InverseGamma(2,3)
  s = 1
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

alg = eNUTS(1000, 0.01)
res = sample(gdemo([1.5, 2.0]), alg)
print(mean(res[:m]))
