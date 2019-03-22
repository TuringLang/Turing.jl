using Turing
using Test
using Random

Random.seed!(125)

# PMMH with Gaussian proposal
GKernel(var) = (x) -> Normal(x, sqrt.(var))
alg = PMMH(1000, SMC(20, :m), MH(1,(:s, GKernel(1))))
chain = sample(gdemo_default, alg)
check_gdemo(chain)
# @test mean(chain[:s].value) ≈ 49/24 atol=0.2
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1

# PMMH with prior as proposal
alg = PMMH(1000, SMC(20, :m), MH(1,:s))
chain = sample(gdemo_default, alg)
check_gdemo(chain)
# @test mean(chain[:s].value) ≈ 49/24 atol=0.1
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1

# PIMH
alg = PIMH(1000, SMC(20))
chain = sample(gdemo_default, alg)
check_gdemo(chain)
# @test mean(chain[:s].value) ≈ 49/24 atol=0.15
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1
