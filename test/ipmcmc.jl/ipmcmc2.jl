using Turing
using Test
using Random

Random.seed!(125)

D = [1.0 1.0 4.0 4.0]

@model MoGtest(D) = begin
  mu1 ~ Normal(1, 1)
  mu2 ~ Normal(4, 1)
  z1 ~ Categorical(2)
  if z1 == 1
    D[1] ~ Normal(mu1, 1)
  else
    D[1] ~ Normal(mu2, 1)
  end
  z2 ~ Categorical(2)
  if z2 == 1
    D[2] ~ Normal(mu1, 1)
  else
    D[2] ~ Normal(mu2, 1)
  end
  z3 ~ Categorical(2)
  if z3 == 1
    D[3] ~ Normal(mu1, 1)
  else
    D[3] ~ Normal(mu2, 1)
  end
  z4 ~ Categorical(2)
  if z4 == 1
    D[4] ~ Normal(mu1, 1)
  else
    D[4] ~ Normal(mu2, 1)
  end
  z1, z2, z3, z4, mu1, mu2
end

gibbs = IPMCMC(15, 100, 10)
chain = sample(MoGtest(D), gibbs)

@test mean(chain[:z1].value) ≈ 1.0 atol=0.1
@test mean(chain[:z2].value) ≈ 1.0 atol=0.1
@test mean(chain[:z3].value) ≈ 2.0 atol=0.1
@test mean(chain[:z4].value) ≈ 2.0 atol=0.1
@test mean(chain[:mu1].value) ≈ 1.0 atol=0.1
@test mean(chain[:mu2].value) ≈ 4.0 atol=0.1
