# Test @assume and @predict macros on a model without conditioning.

using Turing
using Distributions
using Base.Test

@model test_assume() begin
  x ~ Bernoulli(1)
  y ~ Bernoulli(x / 2)
  x, y
end

smc = SMC(10000)
pg = PG(10,1000)

res = sample(test_assume(), smc)

@test reduce(&, res[:x]) == 1 # check that x is always 1
@test_approx_eq_eps mean(res[:y]) 0.5 0.1 # check that the mean of y is between 0.4 and 0.6

res = sample(test_assume(), pg)

@test reduce(&, res[:x]) == 1 # check that x is always 1
@test_approx_eq_eps mean(res[:y]) 0.5 0.1 # check that the mean of y is between 0.4 and 0.6
