# Test @assume and @predict macros on a model without conditioning.

using Turing
using Test

@model test_assume() = begin
  x ~ Bernoulli(1)
  y ~ Bernoulli(x / 2)
  x, y
end

smc = SMC(10000)
pg = PG(10,1000)

res = sample(test_assume(), smc)

@test all(res[:x].value .== 1) # check that x is always 1
@test mean(res[:y].value) ≈ 0.5 atol=0.1 # check that the mean of y is between 0.4 and 0.6

res = sample(test_assume(), pg)

@test all(res[:x].value .== 1) # check that x is always 1
@test mean(res[:y].value) ≈ 0.5 atol=0.1 # check that the mean of y is between 0.4 and 0.6
