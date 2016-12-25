using Distributions
using Turing

@model gauss(data=Dict(:x=>Float64[1 2])) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  @predict s m
  s, m
end

chain = sample(gauss, SMC(10))

chain[:s]
chain[:m]

Turing.TURING[:modelex]
