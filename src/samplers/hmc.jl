immutable HMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
end

type Prior
  prior   :: Function
  dim     :: Int
end

type HMCSampler{HMC} <: Sampler{HMC}
  alg         :: HMC
  model       :: Function
  samples     :: Array{Dict{Symbol, Any}}
  logjoint    :: Dual
  predicts    :: Dict{Symbol, Any}

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Dict{Symbol, Any}}(alg.n_samples)
    for i = 1:alg.n_samples
      samples[i] = Dict{Symbol, Any}()
    end
    logjoint = 0
    predicts = Dict{Symbol, Any}()
    new(alg, model, samples, logjoint, predicts)
  end
end

function Base.run(spl :: Sampler{HMC})
  n = spl.alg.n_samples
  for i = 1:n
    consume(Task(spl.model))
    spl.samples[i] = spl.predicts
    spl.predicts = Dict{Symbol, Any}()
  end
  results = Dict{Symbol, Any}()
  results[:samples] = spl.samples
  return results
end

function assume(spl :: HMCSampler{HMC}, distribution :: Distribution)
  self.logjoint +=
  return rand(distribution)
end

function param(spl :: HMCSampler{HMC}, distribution :: Distribution)
  return assume(spl, distribution)
end

function observe(spl :: HMCSampler{HMC}, score :: Float64)
  print("observing...")
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  spl.predicts[name] = value
end

sample(model :: Function, alg :: HMC) = (
                                        global sampler = HMCSampler{HMC}(alg, model);
                                        run(sampler)
                                       )
