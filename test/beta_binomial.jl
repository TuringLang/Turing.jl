# Test @assume and @predict macros on a model with conditioning.
# We may want to add comparison of p to the test. REVIEW: what does this comment mean? (Kai)

using Turing
using Distributions
using Base.Test

prior = Beta(2,2)
obs = [0,1,0,1,1,1,1,1,1,1]
exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
meanp = exact.α / (exact.α + exact.β)

@model testbb(obs) begin
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

res = @sample(testbb(obs), s)
@test_approx_eq_eps mean(res[:p]) meanp 0.05

res = @sample(testbb(obs), p)
@test_approx_eq_eps mean(res[:x]) meanp 0.10

res = @sample(testbb(obs), g)
@test_approx_eq_eps mean(res[:x]) meanp 0.10
