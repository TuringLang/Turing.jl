using Distributions
using Turing
using Base.Test

x = [1.0 1.0 1.0 4.0 4.0 4.0]

@model MoGtest begin
  N = length(D)
  mu0 ~ Normal(0, 16)
  mu1 ~ Normal(5, 16)
  z = tzeros(N)
  for i in 1:N
    z[i] ~ Bernoulli(0.5)
    if z[i] == 0
      x[i] ~ Normal(mu0, 1)
    else
      x[i] ~ Normal(mu1, 1)
    end
  end
  z, mu0, mu1
end

gibbs = Gibbs(1500, PG(15, 1, :z), HMC(1, 0.25, 3, :mu0, :mu1))
chain = sample(MoGtest, gibbs)

Turing.TURING[:modelex]

mean(chain[:z])
mean(chain[:mu0])
mean(chain[:mu1])

# @test_approx_eq_eps mean(chain[:s]) 49/24 0.15
# @test_approx_eq_eps mean(chain[:m]) 7/6 0.15
