using Turing
using Test

@model fggibbstest(xs) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  # xx ~ Normal(m, sqrt(s)) # this is illegal

  for i = 1:length(xs)
    xs[i] ~ Normal(m, sqrt(s))
  # for xx in xs
    # xx ~ Normal(m, sqrt(s))
  end
  s, m
end

gibbs = Gibbs(2, PG(10, 2, :s), HMC(1, 0.4, 8, :m))
chain = sample(gdemo_default, gibbs);
