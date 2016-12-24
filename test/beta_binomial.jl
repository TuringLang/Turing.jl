# Test @assume and @predict macros on a model with conditioning.
# We may want to add comparison of p to the test. REVIEW: what does this comment mean? (Kai)

using Turing
using Distributions
using Base.Test
using ConjugatePriors: posterior

prior = Beta(2,2)
obs = [0,1,0,1,1,1,1,1,1,1]
exact = posterior(prior, Bernoulli, obs)
meanp = exact.α / (exact.α + exact.β)

@model testbb begin
  p ~ Beta(2,2)
  x ~ Bernoulli(p)
  for i = 1:length(obs)
    obs[i] ~ Bernoulli(p)
  end
end


s = SMC(10000)
p = PG(100,1000)

res = sample(testbb, s)
@test_approx_eq_eps mean(res[:p]) meanp 0.05

res = sample(testbb, p)
@test_approx_eq_eps mean(res[:x]) meanp 0.10
