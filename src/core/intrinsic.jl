abstract InferenceAlgorithm{P}
abstract Sampler{T<:InferenceAlgorithm}

## Fallback functions

Base.run(spl :: Sampler) = error("[sample]: unmanaged inference algorithm: $(typeof(spl))")

assume(spl, distr :: Distribution) =
  error("[assume]: unmanaged inference algorithm: $(typeof(spl))")

observe(spl, weight :: Float64) =
  error("[observe]: unmanaged inference algorithm: $(typeof(spl))")

predict(spl, var_name :: Symbol, value) =
  error("[predict]: unmanaged inference algorithm: $(typeof(spl))")

## Default functions
function sample(model::Function, alg :: InferenceAlgorithm)
  global sampler = ParticleSampler{typeof(alg)}(model, alg);
  Base.run(sampler)
end

assume(spl :: Sampler, dd :: dDistribution, p)  = rand( current_trace(), dd.d )
observe(spl :: Sampler, dd :: dDistribution, value) = produce(logpdf(dd.d, value))

function predict(spl :: Sampler, v_name :: Symbol, value)
  task = current_task()
  if haskey(task.storage, :turing_predicts)
    predicts = task.storage[:turing_predicts]
  else
    predicts = Dict{Symbol,Any}()
  end
  predicts[v_name] = value
  task.storage[:turing_predicts] = predicts
end
