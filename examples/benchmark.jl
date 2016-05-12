using Turing

# Benchmarking function runs an inference algorithm, measures execution time and evaluates the results.
# An optional warmup run is performed to exclude compilation time from measurements.
function benchmark(modelname :: AbstractString, alg :: InferenceAlgorithm, do_eval = true, do_warmup = true)
  # model definition
  include(string(modelname, ".jl"))

  # extract model and evaluation function
  model = eval(symbol(modelname))
  evaluate = eval(symbol(string(modelname,"_evaluate")))

  # warmup
  if do_warmup
    sample(model, alg)
  end

  # proper run with time measurement
  tic()
  chain = sample(model,alg)
  t = toq()

  # compute results
  if do_eval
    results = evaluate(chain)
  else
    results = Dict{Symbol,Any}()
  end
  results[:model] = modelname
  results[:time] = t

  return results
end
