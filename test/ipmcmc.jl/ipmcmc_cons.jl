using Turing, Distributions
using Base.Test

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

N = 50
s1 = IPMCMC(10, N, 4, 2)
s2 = IPMCMC(10, N, 4)

c1 = sample(gdemo(), s1)
c2 = sample(gdemo(), s2)

# Very loose bound, only for testing constructor.
for c in [c1, c2]
  check_numerical(c, [:s, :m], [49/24, 7/6], eps=1.0)
end
