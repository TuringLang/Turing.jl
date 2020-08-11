using Turing
using Test

@model testassume begin
  x ~ Bernoulli(1; :static = true)
  y ~ Bernoulli(x / 2; :param = true)
  1 ~ Normal(0, 1; :static = true)
  2 ~ Normal(0, 1; :param = true)
  y, x
end

s = SMC()

res = sample(testassume, s)

# check that x is always 1
@test all(isone, res[:x])

# check that the mean of y is between 0.4 and 0.6
@test mean(res[:y]) ≈ 0.5 atol=0.1
