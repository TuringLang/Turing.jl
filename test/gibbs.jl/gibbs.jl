using Distributions
using Turing
using Base.Test

srand(125)

x = [1.5 2.0]

@model gibbstest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt.(s))
  end
  s, m
end

alg = Gibbs(3000, CSMC(15, 1, :s), HMC(1, 0.2, 4, :m))
chain = sample(gibbstest(x), alg)
@test mean(chain[:s]) ≈ 49/24 atol=0.1
println("mean(chain[:m]) = $(mean(chain[:m]))")
@test mean(chain[:m]) ≈ 7/6 atol=0.1

alg = CSMC(15, 5000)
chain = sample(gibbstest(x), alg)
@test mean(chain[:s]) ≈ 49/24 atol=0.1
@test mean(chain[:m]) ≈ 7/6 atol=0.1
