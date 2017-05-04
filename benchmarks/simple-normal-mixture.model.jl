# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/basic_estimators/normal_mixture.stan

using StatsFuns
using ForwardDiff: Dual

@model nmmodel(y) = begin
  N = length(y)

  theta ~ Uniform(0, 1)

  mu = tzeros(Dual, 2)
  for i = 1:2
    mu[i] ~ Normal(0, 10)
  end

  k = tzeros(Int64, N)
  for i = 1:N
    k[i] ~ Categorical([theta, 1.0 - theta])
  end

  for i = 1:N
    y[i] ~ Normal(mu[k[i]], 1.0)
  end

  # logtheta_p = map(yᵢ -> [log(theta) + logpdf(Normal(mu[1], 1.0), yᵢ), log(1 - theta) + logpdf(Normal(mu[2], 1.0), yᵢ)], y)
  # map!(logtheta_pᵢ -> logtheta_pᵢ - logsumexp(logtheta_pᵢ), logtheta_p)   # normalization
  # for i = 1:N
  #   k[i] ~ Categorical(exp(logtheta_p[i]))
  # end
end
