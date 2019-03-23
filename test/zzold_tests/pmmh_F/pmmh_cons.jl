using Turing
using Test
using Random

Random.seed!(125)

N = 500
s1 = PMMH(N, SMC(10, :s), MH(1,(:m, (s) -> Normal(s, sqrt(1)))))
s2 = PMMH(N, SMC(10, :s), MH(1,:m))
s3 = PIMH(N, SMC(10))

c1 = sample(gdemo_default, s1)
c2 = sample(gdemo_default, s2)
c3 = sample(gdemo_default, s3)

# Very loose bound, only for testing constructor.
for c in [c1, c2, c3]
    check_gdemo(c, eps=1.0)
  # @test mean(c[:s].value) ≈ 49/24 atol=1.0
  # @test mean(c[:m].value) ≈ 7/6 atol=1.0
end
