using Turing, Distributions
using Base.Test

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

N = 500
s1 = PMMH(N, SMC(10, :s), :m)

c1 = sample(gdemo(), s1)

# Very loose bound, only for testing constructor.
for c in [c1]
  check_numerical(c, [:s, :m], [49/24, 7/6], eps=1.0)
end
