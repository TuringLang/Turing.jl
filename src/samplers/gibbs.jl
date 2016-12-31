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

  function GibbsSampler(model::Function, gibbs::Gibbs)
    n_samplers = length(gibbs.algs)
    samplers = Array{Sampler}(n_samplers)
    for i in 1:n_samplers
      alg = gibbs.algs[i]
      if isa(alg, HMC)
        samplers[i] = HMCSampler{HMC}(alg)
      elseif isa(alg, PG)
        samplers[i] = ParticleSampler{PG}(model, alg)
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

      if isa(sampler, Sampler{HMC})
        is_accept_this, varInfo = step(model, data, sampler, varInfo, i==1)
        is_accept = is_accept_this && is_accept
      elseif isa(sampler, Sampler{PG})
        global sampler = sampler
        ref_particle, _ = step(sampler, ref_particle)
      end

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
  println("[Gibbs]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function sample(model::Function, data::Dict, gibbs::Gibbs)
  global sampler = GibbsSampler{Gibbs}(model, gibbs);
  run(model, data, sampler)
end

function sample(model::Function, gibbs::Gibbs)
  global sampler = GibbsSampler{Gibbs}(model, gibbs);
  run(model, Dict(), sampler)
end
