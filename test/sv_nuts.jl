using Distributions
using Turing
include("sv_helper.jl")

y = readsvdata()

# Stochastic volatility (SV)
@model sv_nuts(y, N) = begin
  τ ~ Exponential(1/100)
  ν ~ Exponential(1/100)
  s = Vector{Real}(N)
  s[1] ~ Exponential(1/100)
  for i = 2:N
    s[i] ~ Normal(log(s[i-1]), τ)
    dy = log(y[i] / y[i-1]) / s[i]
    dy ~ ???  # TODO: figure out what to put here
  end
end

N = length(y)
chain = sample(sv_nuts(y, N), NUTS(1000, 0.65))

describe(chain)
