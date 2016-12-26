using Distributions
using Turing
using Base.Test

@model gibbstest begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  @predict s m
  s, m
end

gibbs = Gibbs(100, HMC(100, 0.1, 5, :s), HMC(100, 0.2, 3, :m))
sample(gibbstest, Dict(:x=>[1.5 2.0]), gibbs)
