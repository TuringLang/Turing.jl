using Distributions
using Turing
using Turing: invlogit

# Load data
# NOTE: put the data file in the same path of this file
readlrdata() = begin
  x = Matrix{Float64}(0, 24)
  y = Vector{Float64}()
  cd(dirname(@__FILE__))
  open("lr_nuts.data") do f
    while !eof(f)
      raw_line = readline(f)
      data_str = filter(str -> length(str) > 0, split(raw_line, r"[ ]+")[1:end-1])
      data = map(str -> parse(str), data_str)
      x = cat(1, x, data[1:end-1]')
      y = cat(1, y, data[end])
    end
  end
  x, y
end

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
