using Test, Random, Distributions, Turing

Random.seed!(125)

alg = SGHMC(10000, 0.02, 0.5)
chain = sample(gdemo_default, alg)
check_gdemo(chain, eps=0.1)
# @test mean(chain[:s].value) ≈ 49/24 atol=0.1
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1
