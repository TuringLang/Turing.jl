immutable Gibbs <: InferenceAlgorithm
  n_iters ::  Int
  algs    ::  Tuple
  Gibbs(n_iters::Int, algs...) = new(n_iters, algs)
end

type GibbsSampler{Gibbs} <: Sampler{Gibbs}
  gibbs       ::  Gibbs               # the sampling algorithm
  samplers    ::  Array{Sampler}      # samplers
  samples     ::  Array{Sample}       # samples
  predicts    ::  Dict{Symbol, Any}   # outputs

  function GibbsSampler(gibbs::Gibbs)
    n_samplers = length(gibbs.algs)
    samplers = Array{Sampler}(n_samplers)
    for i in 1:n_samplers
      alg = gibbs.algs[i]
      if isa(alg, HMC)
        samplers[i] = HMCSampler{HMC}(alg)
      end
    end

    samples = Array{Sample}(gibbs.n_iters)
    weight = 1 / gibbs.n_iters
    for i = 1:gibbs.n_iters
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    predicts = Dict{Symbol, Any}()
    new(gibbs, samplers, samples, predicts)
  end
end

function sample(model::Function, data::Dict, gibbs::Gibbs)
  sampler = GibbsSampler{Gibbs}(gibbs);
  # run(model, data, sampler)
end
