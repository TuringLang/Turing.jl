using Turing, Test

@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

mf = gdemo(1.5, 2.0)

chn = sample(mf, NUTS(2000, 200, 0.6), implementation=:DynamicHMC);

@test mean(chn[:s]) ≈ 49/24 atol=0.2
@test mean(chn[:m]) ≈ 7/6 atol=0.2
