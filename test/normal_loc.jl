# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_loc.stan

using Distributions
using Turing
using Base.Test

data = Dict(:y=>[2.0, 1.0, -0.5, 3.0, 0.25])

@model normal_loc begin
  mu ~ Uniform(-10, 10)
  for n = 1:5
    y[n] ~ Normal(mu, 1.0)
  end
  mu
end

chain = sample(normal_loc, data, HMC(1000, 0.05, 3))
chain[:mu]
