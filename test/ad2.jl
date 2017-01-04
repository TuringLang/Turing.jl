using Distributions
using Turing
using Base.Test

# Define model
@model ad_test2 begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  s, m
end

# Run HMC with chunk_size=1
chain = sample(ad_test2, Dict(:x=>[1.5 2.0]), HMC(100, 0.1, 1), Int64(1))
