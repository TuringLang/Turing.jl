# Test @assume and macros on a model with conditioning.

using Turing
using Distributions
using Base.Test

@model testnoparam begin
  x ~ Bernoulli(1)
  1 ~ Bernoulli(x / 2)
  0 ~ Bernoulli(x / 2)
  x
end

s = SMC(1000)
p = PG(100,10)

res = sample(testnoparam, s)

@test reduce(&, res[:x]) == 1  # check that x is always 1
@test res[:logevidence] ≈ 2 * log(0.5)


res = sample(testnoparam, p)

@test reduce(&, res[:x]) == 1  # check that x is always 1
# PG does not provide logevidence estimate

if isdefined(:SMC2)

  s2 = SMC2(10,100)
  res = sample(test, s2)

  @test reduce(&, res[:x]) == 1  # check that x is always 1
  @test res[:logevidence] ≈ 2 * log(0.5)

end
