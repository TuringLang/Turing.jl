using Turing
using Test
using Random

Random.seed!(125)

N = 50
s1 = IPMCMC(10, N, 4, 2)
s2 = IPMCMC(10, N, 4)

c1 = sample(gdemo_default, s1)
c2 = sample(gdemo_default, s2)

# Very loose bound, only for testing constructor.
for c in [c1, c2]
    check_numerical(c, [:s, :m], [49/24, 7/6], eps=1.0)
  # @test mean(c[:s].value) ≈ 49/24 atol=1.0
  # @test mean(c[:m].value) ≈ 7/6 atol=1.0
end
