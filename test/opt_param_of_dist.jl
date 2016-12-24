using Turing
using Distributions
using Base.Test

@model testassume begin
  x ~ Bernoulli(1; :static = true)
  y ~ Bernoulli(x / 2; :param = true)
  1 ~ Normal(0, 1; :static = true)
  2 ~ Normal(0, 1; :param = true)
end

s = SMC(1000)

res = sample(testassume, s)

@test reduce(&, res[:x]) == 1 # check that x is always 1

@test_approx_eq_eps mean(res[:y]) 0.5 0.1 # check that the mean of y is between 0.4 and 0.6
