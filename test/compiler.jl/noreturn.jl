include("../utility.jl")
using Distributions
using Turing
using Test

xnoreturn = [1.5 2.0]

@model noreturn(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt.(s))
  end
end

chain = sample(noreturn(xnoreturn), HMC(3000, 0.15, 6))

check_numerical(chain, [:s, :m], [49/24, 7/6])
