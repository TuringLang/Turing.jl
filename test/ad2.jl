using Distributions
using Turing
using Base.Test

# Define model
@model ad_test2 begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

# Run HMC with chunk_size=1
chain = sample(ad_test2, HMC(100, 0.1, 1); chunk_size=1)
