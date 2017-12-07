using Turing
using Base.Test
srand(100)

x = [1.5 2.0]

@model smctest(x) = begin
  m ~ Normal(1, 1)
  for i in 1:length(x)
    x[i] ~ Normal(m, 1)
  end
  m
end

results = sample(smctest(x), SMC(100))
m = mean(results[:m])

@test abs(m - 3/2) < 0.1
