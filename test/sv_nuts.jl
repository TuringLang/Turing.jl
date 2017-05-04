using Distributions
using Turing
include("sv_helper.jl")

y = readsvdata()

# Stochastic volatility (SV)
@model sv_nuts(y, N, dy) = begin
  τ ~ Exponential(1/100)
  ν ~ Exponential(1/100)
  s = TArray{Real}(N)
  s[1] ~ Exponential(1/100)
  for i = 2:N
    s[i] ~ Normal(log(s[i-1]), τ)
    s[i] = exp(s[i])
    dy = log(y[i] / y[i-1]) / s[i]
    dy ~ TDist(ν)
  end
end

N = length(y)
chain = sample(sv_nuts(y, N, NaN), NUTS(1000, 0.65))
# chain = sample(sv_nuts(y, N, NaN), Gibbs(100, HMC(2, 0.002, 2, :τ, :ν), PG(50, 2, :s)))

describe(chain)
