using Turing, Distributions, ConjugatePriors

@model gaussdemo begin
  # Define a simple Normal model with unknown mean and variance.
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

chain = sample(gaussdemo, IS(50))
