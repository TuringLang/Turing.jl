using Turing, Distributions
using ForwardDiff: Dual

# Generate synthesised data
xs = rand(Normal(0.5, 1), 100)

# Define model
@model priorsinarray begin
  priors = Vector{Dual}(2)
  @assume priors[1] ~ InverseGamma(2, 3)
  @assume priors[2] ~ Normal(0, sqrt(priors[1]))
  for x in xs
    @observe x ~ Normal(priors[2], sqrt(priors[1]))
  end
  @predict priors
end

chain = sample(priorsinarray, HMC(10, 0.01, 10))
