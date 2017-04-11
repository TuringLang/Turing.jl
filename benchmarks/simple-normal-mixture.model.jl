# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_mixture.stan

@model nmmodel(y) = begin
  N = length(y)
  mu = tzeros(Dual, 2)
  k = tzeros(Int64, N)
  theta ~ Uniform(0, 1)
  for i = 1:2
    mu[i] ~ Normal(0, 10)
  end
  for i = 1:N
    k[i] ~ Categorical([theta, 1.0 - theta])
    y[i] ~ Normal(mu[k[i]], 1.0)
  end
end
