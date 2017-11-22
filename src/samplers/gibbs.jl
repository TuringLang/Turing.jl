doc"""
    Gibbs(n_iters, alg_1, alg_2)

Compositional MCMC interface.

Usage:

```julia
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```
"""
immutable Gibbs <: InferenceAlgorithm
  n_iters   ::  Int     # number of Gibbs iterations
  algs      ::  Tuple   # component sampling algorithms
  thin      ::  Bool    # if thinning to output only after a whole Gibbs sweep
  gid       ::  Int
  Gibbs(n_iters::Int, algs...; thin=true) = new(n_iters, algs, thin, 0)
  Gibbs(alg::Gibbs, new_gid) = new(alg.n_iters, alg.algs, alg.thin, new_gid)
end

const GibbsComponent = Union{Hamiltonian,MH,PG}

function Sampler(alg::Gibbs)
  n_samplers = length(alg.algs)
  samplers = Array{Sampler}(n_samplers)

  space = Set{Symbol}()

  for i in 1:n_samplers
    sub_alg = alg.algs[i]
    if isa(sub_alg, GibbsComponent)
      samplers[i] = Sampler(typeof(sub_alg)(sub_alg, i))
    else
      error("[Gibbs] unsupport base sampling algorithm $alg")
    end
    space = union(space, sub_alg.space)
  end

  # Sanity check for space
  @assert issubset(Turing._compiler_[:pvars], space) "[Gibbs] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

  if Turing._compiler_[:pvars] != space
    warn("[Gibbs] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
  end

  info = Dict{Symbol, Any}()
  info[:samplers] = samplers

  Sampler(alg, info)
end

sample(model::Function, alg::Gibbs;
       save_state=false,         # flag for state saving
       resume_from=nothing,      # chain to continue
       reuse_spl_n=0             # flag for spl re-using
      ) = begin

  # Init the (master) Gibbs sampler
  spl = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        Sampler(alg)

  @assert typeof(spl.alg) == typeof(alg) "[Turing] alg type mismatch; please use resume() to re-use spl"

  # Initialize samples
  sub_sample_n = []
  for sub_alg in alg.algs
    if isa(sub_alg, GibbsComponent)
      push!(sub_sample_n, sub_alg.n_iters)
    else
      error("[Gibbs] unsupport base sampling algorithm $alg")
    end
  end

  # Compute the number of samples to store
  n = reuse_spl_n > 0 ?
      reuse_spl_n :
      alg.n_iters
  sample_n = n * (alg.thin ? 1 : sum(sub_sample_n))

  # Init samples
  time_total = zero(Float64)
  samples = Array{Sample}(sample_n)
  weight = 1 / sample_n
  for i = 1:sample_n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

  # Init parameters
  varInfo = resume_from == nothing ?
            Base.invokelatest(model, VarInfo(), nothing) :
            resume_from.info[:vi]
  n = spl.alg.n_iters; i_thin = 1

  # Gibbs steps
  if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[Gibbs] Sampling...", 0) end
  for i = 1:n
    dprintln(2, "Gibbs stepping...")

    time_elapsed = zero(Float64)
    lp = nothing; epsilon = nothing; lf_num = nothing

    for local_spl in spl.info[:samplers]
      last_spl = local_spl
      # if PROGRESS && haskey(spl.info, :progress) local_spl.info[:progress] = spl.info[:progress] end

      dprintln(2, "$(typeof(local_spl)) stepping...")

      if isa(local_spl.alg, GibbsComponent)
        if isa(local_spl.alg, Hamiltonian)  # clean cache
          local_spl.info[:grad_cache] = Dict{UInt64,Vector}()
        end

        for _ = 1:local_spl.alg.n_iters
          dprintln(2, "recording old θ...")
          time_elapsed_thin = @elapsed varInfo = step(model, local_spl, varInfo, i==1)

          if ~spl.alg.thin
            samples[i_thin].value = Sample(varInfo).value
            samples[i_thin].value[:elapsed] = time_elapsed_thin
            if ~isa(local_spl.alg, Hamiltonian)
              # If statement below is true if there is a HMC component which provides lp and epsilon
              if lp != nothing samples[i_thin].value[:lp] = lp end
              if epsilon != nothing samples[i_thin].value[:epsilon] = epsilon end
              if lf_num != nothing samples[i_thin].value[:lf_num] = lf_num end
            end
            i_thin += 1
          end
          time_elapsed += time_elapsed_thin
        end

        if isa(local_spl.alg, Hamiltonian)
          lp = realpart(getlogp(varInfo))
          epsilon = local_spl.info[:wum][:ϵ][end]
          lf_num = local_spl.info[:lf_num]
        end
      else
        error("[Gibbs] unsupport base sampler $local_spl")
      end
    end

    time_total += time_elapsed

    if spl.alg.thin
      samples[i].value = Sample(varInfo).value
      samples[i].value[:elapsed] = time_elapsed
      # If statement below is true if there is a HMC component which provides lp and epsilon
      if lp != nothing samples[i].value[:lp] = lp end
      if epsilon != nothing samples[i].value[:epsilon] = epsilon end
      if lf_num != nothing samples[i].value[:lf_num] = lf_num end
    end

    if PROGRESS
      haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
    end
  end

  println("[Gibbs] Finished with")
  println("  Running time    = $time_total;")

  if resume_from != nothing   # concat samples
    unshift!(samples, resume_from.value2...)
  end
  c = Chain(0, samples)       # wrap the result by Chain

  if save_state               # save state
    save!(c, spl, model, varInfo)
  end

  c
end
