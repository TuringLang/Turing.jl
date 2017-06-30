using Distributions
using Turing
using Turing: invlogit
using Mamba: describe

include("lr_helper.jl")

x, y = readlrdata()

logistic(x::Real) = invlogit(x)

# Bayesian logistic regression (LR)
@model lr_nuts(x, y, d, n, σ²) = begin
  α ~ Normal(0, σ²)
  β ~ MvNormal(zeros(d), σ² * ones(d))
  for i = 1:n
    y′ = logistic(α + dot(x[i,:], β))
    y[i] ~ Bernoulli(y′)
  end
end

n, d = size(x)
chain = sample(lr_nuts(x, y, d, n, 100), NUTS(1000, 0.65))

describe(chain)
