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

gibbs = Gibbs(4000, PG(100, 1, :s), HMC(1, 0.2, 3, :m))
chain = sample(gibbstest, gibbs)

Turing.TURING[:modelex]
@test_approx_eq_eps mean(chain[:s]) 49/24 0.15
@test_approx_eq_eps mean(chain[:m]) 7/6 0.15
