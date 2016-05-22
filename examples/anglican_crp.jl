## CRP example from Anglican
# https://github.com/probprog/anglican-examples/blob/master/worksheets/crp_aistats.clj

using Turing
using Distributions
using Distances

include("utils.jl")

obs = [10, 11, 12, -100, -150, -200, 0.001, 0.01, 0.005, 0.0]
alpha = 1.72
mu    = 0.0
beta  = 100.0
a     = 1.0
b     = 10.0

#Exact posterior over the number of clusters, taken from the Anglican repo
anglican_crp_exact =
  Categorical(normalize!(map(x -> exp(x),
    [-11.4681, -1.0437, -0.9126, -1.6553, -3.0348,
     -4.9985, -7.5829, -10.9459, -15.6461, -21.6521])))

@model anglican_crp begin
  precision_prior = Gamma(a,b)
  cluster_gen = PolyaUrn(alpha)
  clusters = TArray{Distribution,1}(0)
  assignments = TArray{Any}(length(obs))
  for i = 1:length(obs)
    c = randclass(cluster_gen)
    assignments[i] = c
    if c <= length(clusters)
      likelihood = clusters[c]
    else
      @assert c == length(clusters) + 1
      @assume l ~ precision_prior
      s = sqrt(1 / (beta * l))
      @assume m ~ Normal(mu,s)
      likelihood = Normal(m, 1 / sqrt(l))
      push!(clusters, likelihood)
    end
    @observe obs[i] ~ likelihood
  end
  num_clusters = length(clusters)
  @predict assignments num_clusters clusters
end

function anglican_crp_evaluate(results)
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:num_clusters],  results.value)

  fitted = fit_mle(Categorical, samples, weights)
  KL = kl(fitted, anglican_crp_exact)

  summary = Dict{Symbol,Any}()
  summary[:exact_num_clusters] = anglican_crp_exact
  summary[:fitted_num_clusters] = fitted
  summary[:KL] = KL
  return summary
end


