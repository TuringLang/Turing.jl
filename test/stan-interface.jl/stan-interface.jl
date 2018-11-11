using CmdStan, Turing 

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

model_f = gdemo([1.5, 2.0])

chn = sample(model_f, 2000, 1000, false, 1, CmdStan.Adapt(), CmdStan.Hmc())
