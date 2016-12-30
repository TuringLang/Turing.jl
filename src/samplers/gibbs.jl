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

function Base.run(model, data, spl::Sampler{Gibbs})
  # initialization
  task = current_task()
  n =  spl.gibbs.n_iters
  t_start = time()  # record the start time of HMC
  accept_num = 0    # record the accept number
  varInfo = VarInfo()
  ref_particle = nothing

  # HMC steps
  for i = 1:n
    dprintln(2, "recording old θ...")
    old_values = deepcopy(varInfo.values)
    dprintln(2, "Gibbs stepping...")
    is_accept = true
    for sampler in spl.samplers
      dprintln(2, "$sampler stepping...")
      
      if isa(spl, Sampler{HMC})
        is_accept_this, varInfo = step(model, data, sampler, varInfo, i==1)
      elseif isa(spl, Sampler{PG})
        ref_particle, _ = step(spl, ref_particle)
        is_accept_this = true
      end

      is_accept = is_accept_this && is_accept
      if ~is_accept break end     # if one of the step is reject, reject all
    end
    if is_accept  # accepted => store the new predcits
      spl.samples[i].value = deepcopy(task.storage[:turing_predicts])
      accept_num = accept_num + 1
    else          # rejected => store the previous predcits
      varInfo.values = old_values
      spl.samples[i] = spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function sample(model::Function, data::Dict, gibbs::Gibbs)
  sampler = GibbsSampler{Gibbs}(gibbs);
  run(model, data, sampler)
end
