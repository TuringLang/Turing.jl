using Distributions
using Turing
using Base.Test

D = [1.0 1.0 4.0 4.0]

@model MoGtest begin
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

gibbs = Gibbs(1500, PG(15, 1, :z1, :z2, :z3, :z4), HMC(1, 0.25, 3, :mu1, :mu2))
chain = sample(MoGtest, gibbs)

Turing.TURING[:modelex]

mean(chain[:z1])
mean(chain[:z2])
mean(chain[:z3])
mean(chain[:z4])
mean(chain[:mu1])
mean(chain[:mu2])

# @test_approx_eq_eps mean(chain[:s]) 49/24 0.15
# @test_approx_eq_eps mean(chain[:m]) 7/6 0.15
