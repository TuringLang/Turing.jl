## Gaussian example from Anglican
# https://github.com/probprog/anglican-examples/blob/master/worksheets/gaussian_aistats.clj

using Turing
using Distributions
using ConjugatePriors
import Distributions:
  NormalKnownSigma

function kl(p::Normal, q::Normal)
  return (log(q.σ / p.σ) + (p.σ^2 + (p.μ - q.μ)^2) / (2 * q.σ^2) - 0.5)
end

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

# KL-divergence between a normal distribution MLE-fitted to samples
# and true posterior.
function anglican_gaussian_divergence(samples, weights)
  g = NormalKnownSigma(s)
  ss = suffstats(g, samples, weights)
  fitted = fit_mle(g, ss)
  return kl(fitted, anglican_gaussian_exact)
end

function anglican_gaussian_divergence(samples :: Vector{Float64})
  weights = fill(Float64(1), length(samples))
  anglican_gaussian_divergence(samples, weights)
end

function anglican_gaussian_divergence(results)
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:mean],  results.value)
  return anglican_gaussian_divergence(samples, weights)
end
