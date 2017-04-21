immutable Gibbs <: InferenceAlgorithm
  n_iters   ::  Int     # number of Gibbs iterations
  algs      ::  Tuple   # component sampling algorithms
  thin      ::  Bool    # if thinning to output only after a whole Gibbs sweep
  group_id  ::  Int
  Gibbs(n_iters::Int, algs...; thin=true) = new(n_iters, algs, thin, 0)
  Gibbs(alg::Gibbs, new_group_id) = new(alg.n_iters, alg.algs, alg.thin, new_group_id)
end

type GibbsSampler{Gibbs} <: Sampler{Gibbs}
  gibbs       ::  Gibbs               # the sampling algorithm
  samplers    ::  Array{Sampler}      # samplers
  samples     ::  Array{Sample}       # samples

  function GibbsSampler(gibbs::Gibbs)
    n_samplers = length(gibbs.algs)
    samplers = Array{Sampler}(n_samplers)

    space = Set{Symbol}()

    sub_sample_n = []   # record #samples for each sampler
    for i in 1:n_samplers
      alg = gibbs.algs[i]
      if isa(alg, HMC) || isa(alg, HMCDA)
        samplers[i] = HMCSampler{typeof(alg)}(typeof(alg)(alg, i))
        push!(sub_sample_n, alg.n_samples)
      elseif isa(alg, PG)
        samplers[i] = ParticleSampler{PG}(PG(alg, i))
        push!(sub_sample_n, alg.n_iterations)
      else
        error("[GibbsSampler] unsupport base sampling algorithm $alg")
      end
      space = union(space, alg.space)
    end

    # Sanity check for space
    @assert issubset(Turing._compiler_[:pvars], space) "[GibbsSampler] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

    if Turing._compiler_[:pvars] != space
      warn("[GibbsSampler] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
    end

    # Compute the number of samples to store
    if gibbs.thin
      sample_n = gibbs.n_iters
    else
      sample_n = gibbs.n_iters * sum(sub_sample_n)
    end

    # Initialize samples
    samples = Array{Sample}(sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    new(gibbs, samplers, samples)
  end
end

function sample(model::Function, gibbs::Gibbs)
  spl = GibbsSampler{Gibbs}(gibbs);
  n = spl.gibbs.n_iters
  # initialization
  varInfo = model()
  ref_particle = nothing
  i_thin = 1

  # Gibbs steps
  @showprogress 1 "[Gibbs] Sampling..." for i = 1:n
    dprintln(2, "Gibbs stepping...")

    for local_spl in spl.samplers
      # dprintln(2, "Sampler stepping...")
      dprintln(2, "$(typeof(local_spl)) stepping...")
      # println(varInfo)
      if isa(local_spl, Sampler{HMC}) || isa(local_spl, Sampler{HMCDA})

        for _ = 1:local_spl.alg.n_samples
          dprintln(2, "recording old θ...")
          old_vals = deepcopy(varInfo.vals)
          is_accept, varInfo = step(model, local_spl, varInfo, i==1)
          if ~is_accept
            # NOTE: this might cause problem if new variables is added to VarInfo,
            #    which will add new elements to vi.idcs etc.
            varInfo.vals = old_vals
          end
          if ~spl.gibbs.thin
            spl.samples[i_thin].value = Sample(varInfo).value
            i_thin += 1
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
        varInfo = retain(deepcopy(varInfo), local_spl.alg.group_id, 0, local_spl)
        # Local samples
        for _ = 1:local_spl.alg.n_iterations
          ref_particle, samples = step(model, local_spl, varInfo, ref_particle)
          if ~spl.gibbs.thin
            spl.samples[i_thin].value = Sample(ref_particle.vi).value
            i_thin += 1
          end
        end
        varInfo = ref_particle.vi
      end

    end
    if spl.gibbs.thin
      spl.samples[i].value = Sample(varInfo).value
    end

  end

  Chain(0, spl.samples)    # wrap the result by Chain
end
