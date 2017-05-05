using Distributions
using Turing
using Base.Test

x = [1.5 2.0]

@model gibbstest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

check_numerical(
  sample(gibbstest(x), Gibbs(1500, PG(30, 3, :s), HMC(1, 0.2, 4, :m))),
  [:s, :m], [49/24, 7/6]
)

check_numerical(
  sample(gibbstest(x), PG(30, 2500)),
  [:s, :m], [49/24, 7/6]
)
