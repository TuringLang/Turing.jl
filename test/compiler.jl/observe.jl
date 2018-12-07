# Test @assume and macros on a model with conditioning.

using Turing
using Test

@model test() = begin
  z ~ Normal(0,1)
  x ~ Bernoulli(1)
  1 ~ Bernoulli(x / 2)
  0 ~ Bernoulli(x / 2)
  x
end

is  = IS(10000)
smc = SMC(10000)
pg  = PG(100,10)

res = sample(test(), is)

@test reduce(&, res[:x]) == 1  #c heck that x is always 1
@test first(res[:logevidence]) ≈ 2 * log(0.5)

res = sample(test(), smc)

@test reduce(&, res[:x]) == 1  #c heck that x is always 1
@test first(res[:logevidence]) ≈ 2 * log(0.5)


res = sample(test(), pg)

@test reduce(&, res[:x]) == 1  # check that x is always 1
# PG does not provide logevidence estimate
