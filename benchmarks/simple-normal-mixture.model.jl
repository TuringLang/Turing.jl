# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_mixture.stan

@model nmmodel(y) = begin
  N = length(y)
  mu = tzeros(Dual, 2)
  k = tzeros(Int64, N)
  theta ~ Uniform(0, 1)
  for i = 1:2
    mu[i] ~ Normal(0, 10)
  end
  theta_p = map(yᵢ -> [theta * pdf(Normal(mu[1], 1.0), yᵢ), (1 - theta) * pdf(Normal(mu[2], 1.0), yᵢ)], y)
  map!(theta_pᵢ -> theta_pᵢ / sum(theta_pᵢ), theta_p)   # normalization
  for i = 1:N
    k[i] ~ Categorical(theta_p[i])
  end
end
