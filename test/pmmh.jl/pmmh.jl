using Turing
using Test
using Random

Random.seed!(125)

x = [1.5 2.0]

@model pmmhtest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

# PMMH with Gaussian proposal
GKernel(var) = (x) -> Normal(x, sqrt.(var))
alg = PMMH(1000, SMC(20, :m), MH(1,(:s, GKernel(1))))
chain = sample(pmmhtest(x), alg)
@test mean(chain[:s]) ≈ 49/24 atol=0.2
@test mean(chain[:m]) ≈ 7/6 atol=0.1

# PMMH with prior as proposal
alg = PMMH(1000, SMC(20, :m), MH(1,:s))
chain = sample(pmmhtest(x), alg)
@test mean(chain[:s]) ≈ 49/24 atol=0.1
@test mean(chain[:m]) ≈ 7/6 atol=0.1

# PIMH
alg = PIMH(1000, SMC(20))
chain = sample(pmmhtest(x), alg)
@test mean(chain[:s]) ≈ 49/24 atol=0.15
@test mean(chain[:m]) ≈ 7/6 atol=0.1
