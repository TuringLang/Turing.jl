using Turing
using Base.Test

# Test the @assume and @predict macros on a model without conditioning.

@model test begin
  @assume x ~ Bernoulli(1)
  @observe 1 ~ Bernoulli(x / 2)
  @observe 0 ~ Bernoulli(x / 2)
  @predict x
end

s = SMC(1000)
p = PG(100,10)

res = sample(test, s)

@test reduce(&, res[:x]) == 1  #check that x is always 1
@test res[:logevidence] ≈ 2 * log(0.5)


res = sample(test, p)

@test reduce(&, res[:x]) == 1  #check that x is always 1
# PG does not provide logevidence estimate

if isdefined(:SMC2)

  s2 = SMC2(10,100)
  res = sample(test, s2)

  @test reduce(&, res[:x]) == 1  #check that x is always 1
  @test res[:logevidence] ≈ 2 * log(0.5)

end
