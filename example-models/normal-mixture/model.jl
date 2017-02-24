# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_mixture.stan

@model nmmodel begin
  theta ~ Uniform(0, 1)
  mu = Array{Any}(2)
  for k = 1:2
    mu[k] ~ Normal(0, 10)
  end
  for n = 1:1000
    k = rand() < theta ? 1 : 2
    y[n] ~ Normal(mu[k], 1.0)
  end
  mu
end
