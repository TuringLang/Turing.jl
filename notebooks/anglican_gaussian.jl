## Gaussian example from Anglican
# https://github.com/probprog/anglican-examples/blob/master/worksheets/gaussian_aistats.clj

using Turing
using Distributions
using ConjugatePriors
import Distributions:
  NormalKnownSigma

m0 = 1
s0 = sqrt(5)
s  = sqrt(2)
obs = [9.0, 8.0]

prior = Normal(m0,s0)
anglican_gaussian_exact = posterior(prior, NormalKnownSigma(s), obs)

@model anglican_gaussian begin
  @assume mean ~ prior
  for i = 1:length(obs)
    @observe obs[i] ~ Normal(mean, s)
  end
  @predict mean
end

function anglican_gaussian_evaluate(results)
  # Weigthed samples of the mean
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:mean],  results.value)

  # Gaussian with the known variance MLE-fitted to the samples
  # g = NormalKnownSigma(s)
  ss = suffstats(Normal, samples, weights)
  fitted = fit_mle(Normal, ss)

  # KL-divergence between distribution fitted above
  # and the true posterior.
  KL = kl(fitted, anglican_gaussian_exact)

  # Aggregating results
  summary = Dict{Symbol,Any}()
  summary[:exact] = anglican_gaussian_exact
  summary[:fitted] = fitted
  summary[:KL] = KL
  return summary
end
