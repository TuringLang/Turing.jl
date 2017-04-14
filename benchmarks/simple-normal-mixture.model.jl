# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_mixture.stan

using StatsFuns

@model nmmodel(y) = begin
  N = length(y)
  mu = tzeros(Dual, 2)
  k = tzeros(Int64, N)
  theta ~ Uniform(0, 1)
  for i = 1:2
    mu[i] ~ Normal(0, 10)
  end
  logtheta_p = map(yᵢ -> [log(theta) + logpdf(Normal(mu[1], 1.0), yᵢ), log(1 - theta) + logpdf(Normal(mu[2], 1.0), yᵢ)], y)
  map!(logtheta_pᵢ -> logtheta_pᵢ - logsumexp(logtheta_pᵢ), logtheta_p)   # normalization
  for i = 1:N
    k[i] ~ Categorical(exp(logtheta_p[i]))
  end
end
