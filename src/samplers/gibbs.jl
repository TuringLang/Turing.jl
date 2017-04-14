immutable Gibbs <: InferenceAlgorithm
  n_iters   ::  Int
  algs      ::  Tuple
  group_id  ::  Int
  Gibbs(n_iters::Int, algs...) = new(n_iters, algs, 0)
  Gibbs(alg::Gibbs, new_group_id) = new(alg.n_iters, alg.algs, new_group_id)
end

type GibbsSampler{Gibbs} <: Sampler{Gibbs}
  gibbs       ::  Gibbs               # the sampling algorithm
  samplers    ::  Array{Sampler}      # samplers
  samples     ::  Array{Sample}       # samples
  predicts    ::  Dict{Symbol, Any}   # outputs

  function GibbsSampler(model::Function, gibbs::Gibbs)
    n_samplers = length(gibbs.algs)
    samplers = Array{Sampler}(n_samplers)

    space = Set{Symbol}()

    for i in 1:n_samplers
      alg = gibbs.algs[i]
      if isa(alg, HMC)
        samplers[i] = HMCSampler{HMC}(HMC(alg, i))
      elseif isa(alg, PG)
        samplers[i] = ParticleSampler{PG}(PG(alg, i))
      else
        error("[GibbsSampler] unsupport base sampling algorithm $alg")
      end
      space = union(space, alg.space)
    end



    @assert issubset(TURING[:model_pvar_list], space) "[GibbsSampler] symbols specified to samplers ($space) doesn't cover the model parameters ($(TURING[:model_pvar_list]))"

    if TURING[:model_pvar_list] != space
      warn("[GibbsSampler] extra parameters specified by samplers don't exist in model: $(setdiff(space, TURING[:model_pvar_list]))")
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
  n = spl.gibbs.n_iters
  varInfo = VarInfo()
  ref_particle = nothing

  # Gibbs steps
  @showprogress 1 "[Gibbs] Sampling..." for i = 1:n
    dprintln(2, "Gibbs stepping...")

    for local_spl in spl.samplers
      # dprintln(2, "Sampler stepping...")
      dprintln(2, "$(typeof(local_spl)) stepping...")
      # println(varInfo)
      if isa(local_spl, Sampler{HMC})

        for _ in local_spl.alg.n_samples
          dprintln(2, "recording old θ...")
          old_vals = deepcopy(varInfo.vals)
          is_accept, varInfo = step(model, local_spl, varInfo, i==1)
          if ~is_accept
            varInfo.vals = old_vals
          end
        end
      elseif isa(local_spl, Sampler{PG})
        # Update new VarInfo to the reference particle
        varInfo.index = 0
        varInfo.num_produce = 0
        if ref_particle != nothing
          ref_particle.vi = varInfo
        end
        # Clean variables belonging to the current sampler
        varInfo = retain(deepcopy(varInfo), local_spl.alg.group_id, 0)
        # Local samples
        for _ in local_spl.alg.n_iterations
          ref_particle, samples = step(model, local_spl, varInfo, ref_particle)
        end
        varInfo = ref_particle.vi
      end
    end
    spl.samples[i].value = Sample(varInfo).value
  end

  Chain(0, spl.samples)    # wrap the result by Chain
end

function sample(model::Function, data::Dict, gibbs::Gibbs)
  global sampler = GibbsSampler{Gibbs}(model, gibbs);
  run(model, data, sampler)
end

function sample(model::Function, gibbs::Gibbs)
  global sampler = GibbsSampler{Gibbs}(model, gibbs);
  run(model, Dict(), sampler)
end
