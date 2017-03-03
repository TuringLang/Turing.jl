using Distributions
using Turing
using Base.Test

xs = [1.5 2.0]
xx = 1

@model fggibbstest(x) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  # xx ~ Normal(m, sqrt(s)) # this is illegal
  for xx in xs
    xx ~ Normal(m, sqrt(s))
  end
  s, m
end

gibbs = Gibbs(2000, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
chain = @sample(fggibbstest(xs), gibbs)
