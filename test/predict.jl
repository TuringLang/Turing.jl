using Turing
using Base.Test

# Test the @predict macro on a deterministic model.

@model test begin
  x = 0
  @predict x
  x = 1
  y = 2
  @predict y
end

s = SMC(10)
p = PG(2,5)

res = sample(test, s)

@test res[:x][1] == 0
@test res[:y][1] == 2


res = sample(test, p)

@test res[:x][1] == 0
@test res[:y][1] == 2
