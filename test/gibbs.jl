using Distributions
using Turing
using Base.Test

x = [1.5 2.0]

@model gibbstest begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

gibbs = Gibbs(2000, PG(0, 10, :s), HMC(0, 0.2, 3, :m))
chain = sample(gibbstest, Dict(:x=>[1.5 2.0]), gibbs)

Turing.TURING[:modelex]
mean(chain[:s]) - 7/6
mean(chain[:m]) - 49/24
