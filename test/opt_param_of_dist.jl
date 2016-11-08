using Turing
using Distributions
using Base.Test

@model testassume begin
  @assume x ~ Bernoulli(1; :static = true)
  @assume y ~ Bernoulli(x / 2; :param = true)
  @observe 1 ~ Normal(0, 1; :static = true)
  @observe 2 ~ Normal(0, 1; :param = true)
  @predict y
  @predict x
end

s = SMC(100)

res = sample(testassume, s)

@test reduce(&, res[:x]) == 1 # check that x is always 1
@test_approx_eq_eps mean(res[:y]) 0.5 0.1 # check that the mean of y is between 0.4 and 0.6
