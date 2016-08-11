## HMM example from Anglican
# https://github.com/probprog/anglican-examples/blob/master/worksheets/hmm_aistats.clj

using Turing
using Distributions
using Distances

statesmean = [-1, 1, 0]
initial    = Categorical([1.0/3, 1.0/3, 1.0/3])
trans      = [Categorical([0.1, 0.5, 0.4]), Categorical([0.2, 0.2, 0.6]), Categorical([0.15, 0.15, 0.7])]
data       = [0, 0.9, 0.8, 0.7, 0, -0.025, -5, -2, -0.1, 0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]

# Matrix of exact marginals for posterior states.
# Copied from the Anglican repo.
anglican_hmm_matrix =   [[ 0.3775 0.3092 0.3133];
                        [ 0.0416 0.4045 0.5539];
                        [ 0.0541 0.2552 0.6907];
                        [ 0.0455 0.2301 0.7244];
                        [ 0.1062 0.1217 0.7721];
                        [ 0.0714 0.1732 0.7554];
                        [ 0.9300 0.0001 0.0699];
                        [ 0.4577 0.0452 0.4971];
                        [ 0.0926 0.2169 0.6905];
                        [ 0.1014 0.1359 0.7626];
                        [ 0.0985 0.1575 0.744 ];
                        [ 0.1781 0.2198 0.6022];
                        [ 0.0000 0.9848 0.0152];
                        [ 0.1130 0.1674 0.7195];
                        [ 0.0557 0.1848 0.7595];
                        [ 0.2017 0.0472 0.7511];
                        [ 0.2545 0.0611 0.6844]]

# Convert anglican_hmm_matrix to a vector of vectors
anglican_hmm_exact = Vector{Categorical}(size(anglican_hmm_matrix)[1])
for i = 1:length(anglican_hmm_exact)
  anglican_hmm_exact[i] = Categorical(normalize!(squeeze(anglican_hmm_matrix[i,:],1)))
end

@model anglican_hmm begin
  states = TArray(Int, length(data))
  @assume(states[1] ~ initial)
  for i = 2:length(data)
    @assume(states[i] ~ trans[states[i-1]])
    @observe(data[i]  ~ Normal(statesmean[states[i]], 1))
  end
  @predict states
end


function anglican_hmm_evaluate(results)
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:states], results.value)
  marginal_samples = Vector{Vector{Int}}(length(data))
  marginal_dists = Vector{Categorical}(length(data))
  for i = 1:length(data)
    marginal_samples[i] = Vector{Int}(length(samples))
    for j = 1:length(samples)
      marginal_samples[i][j] = samples[j][i]
    end
  marginal_dists[i] = fit_mle(Categorical, marginal_samples[i], weights)
  end

  KLs = map(kl, marginal_dists, anglican_hmm_exact)
  KL = sum(KLs)

  summary = Dict{Symbol,Any}()
  summary[:exact_marginals] = anglican_hmm_exact
  summary[:fitted_marginals] = marginal_dists
  summary[:KL] = KL
  return summary
end
