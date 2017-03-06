# Test replay for loop

using Turing, Distributions
using ForwardDiff: Dual

# Generate synthesised data
xs = rand(Normal(0.5, 1), 100)

# Define model
@model priorsinarray(xs) begin
  priors = Vector{Dual}(2)
  priors[1] ~ InverseGamma(2, 3)
  priors[2] ~ Normal(0, sqrt(priors[1]))
  for i = 1:length(xs)
    xs[i] ~ Normal(priors[2], sqrt(priors[1]))
  # for x in xs
    # x ~ Normal(priors[2], sqrt(priors[1]))
  end
  priors
end

# Sampling
chain = @sample(priorsinarray(xs), HMC(10, 0.01, 10))
