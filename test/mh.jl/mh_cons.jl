using Turing
using Test
using Random
Random.seed!(0)

@model gdemo() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

GaussianKernel(var) = (x) -> Normal(x, sqrt(var))
N = 2000
s1 = MH(N, (:s, GaussianKernel(3.0)), (:m, GaussianKernel(3.0)))
s2 = MH(N, :s, :m)
s3 = MH(N)
s4 = Gibbs(N, MH(5, :m), MH(5, :s))

c1 = sample(gdemo(), s1)
c2 = sample(gdemo(), s2)
c3 = sample(gdemo(), s3)
c4 = sample(gdemo(), s4)

# Very loose bound, only for testing constructor.
for c in [c1, c2, c3, c4]
  @test mean(c[:s]) ≈ 49/24 atol=1.0
  @test mean(c[:m]) ≈ 7/6 atol=1.0
end
