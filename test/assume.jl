using Turing
using Distributions
using Base.Test

# Test the @assume and @predict macros on a model without conditioning.

@model testassume begin
  @assume x ~ Bernoulli(1)
  @assume y ~ Bernoulli(x / 2)
  @predict y
  @predict x
end

s = SMC(10000)
p = PG(10,1000)

res = sample(testassume, s)

@test reduce(&, res[:x]) == 1  #check that x is always 1
@test_approx_eq_eps mean(res[:y]) 0.5 0.1  #check that the mean of y is between 0.4 and 0.6


res = sample(testassume, p)

@test reduce(&, res[:x]) == 1  #check that x is always 1
@test_approx_eq_eps mean(res[:y]) 0.5 0.1  #check that the mean of y is between 0.4 and 0.6
