using Distributions
using Turing

@model gauss(data=Dict(:x=>Float64[1 2])) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

gauss()

Turing.TURING[:modelex]
