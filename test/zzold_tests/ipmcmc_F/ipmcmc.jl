using Turing
using Test
using Random

Random.seed!(125)

alg = IPMCMC(30, 500, 4)
chain = sample(gdemo_default, alg)
check_numerical(chain, [:s, :m], [49/24, 7/6])

# @test mean(chain[:s].value) ≈ 49/24 atol=0.1
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1
