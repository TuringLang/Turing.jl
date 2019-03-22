using Turing
using Test
using Random
Random.seed!(125)

# MH with prior as proposal
alg = MH(2000)
chain = sample(gdemo_default, alg)
check_gdemo(chain, eps = 0.1)
# @test mean(chain[:s].value) ≈ 49/24 atol=0.1
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1

# MH with Gaussian proposal
GKernel(var) = (x) -> Normal(x, sqrt.(var))
alg = MH(5000, (:s, GKernel(5)), (:m, GKernel(1.0)))
chain = sample(gdemo_default, alg)
check_gdemo(chain, eps = 0.1)

# @test mean(chain[:s].value) ≈ 49/24 atol=0.2
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1

# MH within Gibbs
alg = Gibbs(1000, MH(5, :m), MH(5, :s))
chain = sample(gdemo_default, alg)
check_gdemo(chain, eps = 0.1)

# @test mean(chain[:s].value) ≈ 49/24 atol=0.1
# @test mean(chain[:m].value) ≈ 7/6 atol=0.1
