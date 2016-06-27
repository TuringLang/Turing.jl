immutable HMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
end

type HMCSampler{HMC} <: Sampler{HMC}
  alg         :: HMC
  model       :: Function
  samples     :: Array{Dict{Symbol, Any}}
  logevidence :: Float64
  predicts    :: Dict{Symbol, Any}
  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Dict{Symbol, Any}}(alg.n_samples)
    for i = 1:alg.n_samples
      samples[i] = Dict{Symbol, Any}()
    end
    logevidence = 0
    predicts = Dict{Symbol, Any}()
    new(alg, model, samples, logweights, logevidence, predicts)
  end
end

function Base.run(spl :: Sampler{HMC})
  n = spl.alg.n_samples
  for i = 1:n
    consume(Task(spl.model))
    spl.samples[i] = spl.predicts
    spl.logevidence = 0
    spl.predicts = Dict{Symbol, Any}()
  end
  spl.logevidence = logsum(spl.logweights) - log(n)
  results = Dict{Symbol, Any}()
  results[:logevidence] = spl.logevidence
  results[:samples] = spl.samples
  return results
end

function assume(spl :: HMCSampler{HMC}, distribution :: Distribution)
  return rand(distribution)
end

function param(spl :: HMCSampler{HMC}, distribution :: Distribution)
  return assume(spl, distribution)
end

function observe(spl :: HMCSampler{HMC}, score :: Float64)
  spl.logevidence += score
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  spl.predicts[name] = value
end

sample(model :: Function, alg :: IS) = (
                                        global sampler = HMCSampler{HMC}(alg, model);
                                        run(sampler)
                                       )
