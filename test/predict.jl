# Test the @predict macro on a deterministic model.

using Turing
using Distributions
using Base.Test

<<<<<<< HEAD
# Test the @predict macro on a deterministic model.

=======
>>>>>>> development
@model testpredict begin
  x = 0
  @predict x
  x = 1
  y = 2
  @predict y
end

s = SMC(10)
p = PG(2,5)

res = sample(testpredict, s)

@test res[:x][1] == 0
@test res[:y][1] == 2


res = sample(testpredict, p)

@test res[:x][1] == 0
@test res[:y][1] == 2
