include("../utility.jl")

using Distributions, Turing
using Mamba: describe

@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

model_f = gdemo([1.5, 2.0])

alg = NUTS(2500, 500, 0.65)
res = sample(model_f, alg)
describe(res)
check_numerical(res, [:s, :m], [49/24, 7/6])


alg = Gibbs(2500, NUTS(500, 0.65, :m), PG(50, 1, :s))
res2 = sample(model_f, alg)

check_numerical(res2, [:s, :m], [49/24, 7/6])
