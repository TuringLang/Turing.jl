using Turing
using Test
using Random
Random.seed!(0)

GaussianKernel(var) = (x) -> Normal(x, sqrt(var))
N = 2000
s1 = MH(N, (:s, GaussianKernel(3.0)), (:m, GaussianKernel(3.0)))
s2 = MH(N, :s, :m)
s3 = MH(N)
s4 = Gibbs(N, MH(5, :m), MH(5, :s))

c1 = sample(gdemo_default, s1)
c2 = sample(gdemo_default, s2)
c3 = sample(gdemo_default, s3)
c4 = sample(gdemo_default, s4)

# Very loose bound, only for testing constructor.
for c in [c1, c2, c3, c4]
    check_gdemo(c, eps = 1.0)
  # @test mean(c[:s].value) ≈ 49/24 atol=1.0
  # @test mean(c[:m].value) ≈ 7/6 atol=1.0
end
