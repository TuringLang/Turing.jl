using Distributions
using Turing
using Turing: invlogit
include("lr_helper.jl")

x, y = readlrdata()

logistic(x::Real) = invlogit(x)

# Bayesian logistic regression (LR)
@model hlr_nuts(x, y, d, n, Θ) = begin
  σ² ~ Exponential(Θ)
  α ~ Normal(0, σ²)
  β ~ MvNormal(zeros(d), σ² * ones(d))
  for i = 1:n
    y′ = logistic(α + dot(x[i,:], β))
    y[i] ~ Bernoulli(y′)
  end
end

n, d = size(x)
λ = 0.01; Θ = 1 \ λ
chain = sample(lr_nuts(x, y, d, n, Θ), NUTS(1000, 0.65))

describe(chain)
