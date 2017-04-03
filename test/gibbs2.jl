using Distributions
using Turing
using Base.Test

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

gibbs = Gibbs(500, PG(10, 1, :z1, :z2, :z3, :z4), HMC(1, 0.15, 3, :mu1, :mu2))
chain = sample(MoGtest(D), gibbs)

# chain = sample(MoGtest, SMC(5000))

Turing.TURING[:modelex]

mean(chain[:z1])
mean(chain[:z2])
mean(chain[:z3])
mean(chain[:z4])
mean(chain[:mu1])
mean(chain[:mu2])
