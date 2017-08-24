using Distributions
using Turing
using Base.Test

srand(125)

x = [1.5 2.0]

@model pmmhtest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

check_numerical(
  sample(pmmhtest(x), PMMH(1500, SMC(30, :s), :m)),
  [:s, :m], [49/24, 7/6]
)
