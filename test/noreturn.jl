using Distributions
using Turing
using Base.Test

x = [1.5 2.0]

@model noreturn begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
end

chain = sample(noreturn, HMC(100, 0.4, 8))
