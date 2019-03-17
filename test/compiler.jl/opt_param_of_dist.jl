using Turing
using Test

@model testassume() = begin
  x ~ Bernoulli(1)
  y ~ Bernoulli(x / 2)
  1 ~ Normal(0, 1)
  2 ~ Normal(0, 1)
  y, x
end

s = SMC(1000)

res = sample(testassume(), s)

@test reduce(&, res[:x].value .== 1) # check that x is always 1

@test mean(res[:y].value) ≈ 0.5 atol=0.1 # check that the mean of y is between 0.4 and 0.6
