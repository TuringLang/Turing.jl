using Turing
using Distributions
using Base.Test

# Test the @assume and @predict macros on a model without conditioning.
# We may want to add comparison of p to the test.

prior = Beta(2,2)
obs = [0,1,0,1,1,1,1,1,1,1]
exact = posterior(prior, Bernoulli, obs)
meanp = exact.α / (exact.α + exact.β)

@model test begin
  @assume p ~ Beta(2,2)
  @assume x ~ Bernoulli(p)
  for i = 1:length(obs)
    @observe obs[i] ~ Bernoulli(p)
  end
  @predict p x
end


s = SMC(10000)
p = PG(100,1000)

res = sample(test, s)
@test_approx_eq_eps mean(res[:p]) meanp 0.05

res = sample(test, p)
@test_approx_eq_eps mean(res[:x]) meanp 0.10
