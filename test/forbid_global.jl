using Distributions
using Turing
using Base.Test

x = [1.5 2.0]
xx = 1

@model gibbstest(x) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  # xx ~ Normal(m, sqrt(s)) # this is illegal
  for xx in x
    xx ~ Normal(m, sqrt(s))
  end
  s, m
end

gibbs = Gibbs(2000, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
chain = @sample(gibbstest(x), gibbs)
