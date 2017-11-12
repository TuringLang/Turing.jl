using Turing, Distributions
using Base.Test
include("../utility.jl")

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  1.5 ~ Normal(m, sqrt.(s))
  2.0 ~ Normal(m, sqrt.(s))
  return s, m
end

N = 500
s1 = PMMH(N, SMC(10, :s), MH(1,(:m, (s) -> Normal(s, sqrt(3.0)))))
s2 = PMMH(N, SMC(10, :s), MH(1,:m))
s3 = PIMH(N, SMC(10))

c1 = sample(gdemo(), s1)
c2 = sample(gdemo(), s2)
c3 = sample(gdemo(), s3)

# Very loose bound, only for testing constructor.
for c in [c1, c2, c3]
  check_numerical(c, [:s, :m], [49/24, 7/6], eps=1.0)
end
