using Distributions
using Turing

x = Float64[1 2]
priors = 0

@model gauss(x) = begin
  priors = TArray{Float64}(2)
  priors[1] ~ InverseGamma(2,3)         # s
  priors[2] ~ Normal(0, sqrt(priors[1])) # m
  for i in 1:length(x)
    x[i] ~ Normal(priors[2], sqrt(priors[1]))
  end
  priors
end

chain = sample(gauss(x), PG(10, 10))
chain = sample(gauss(x), SMC(10))
