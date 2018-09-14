# Test @assume and @predict macros on a model with conditioning.
# We may want to add comparison of p to the test. REVIEW: what does this comment mean? (Kai)

using Turing
using Test

prior = Beta(2,2)
obs = [0,1,0,1,1,1,1,1,1,1]
exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
meanp = exact.α / (exact.α + exact.β)

@model testbb(obs) = begin
  p ~ Beta(2,2)
  x ~ Bernoulli(p)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
  p, x
end


s = SMC(10000)
p = PG(100,1000)
g = Gibbs(1500, HMC(1, 0.2, 3, :p), PG(100, 1, :x))

check_numerical(
  sample(testbb(obs), s),
  [:p], [meanp], eps=0.05
)

check_numerical(
  sample(testbb(obs), p),
  [:x], [meanp], eps=0.1
)

check_numerical(
  sample(testbb(obs), g),
  [:x], [meanp], eps=0.1
)
