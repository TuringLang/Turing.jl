using Distributions
using Turing
using Base.Test

# Define model
@model ad_test begin
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

# Run HMC with chunk_size=1
chain = sample(ad_test, HMC(1, 0.1, 1); chunk_size=1)
