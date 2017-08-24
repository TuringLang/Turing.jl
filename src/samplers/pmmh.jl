doc"""
    PMMH(n_iters::Int, smc_alg:::SMC,)

Particle marginal Metropolis–Hastings sampler.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), :v2)
```
"""
immutable PMMH <: InferenceAlgorithm
  n_iters               ::    Int         # number of iterations
  smc_alg               ::    SMC         # SMC targeting state
  space                 ::    Set         # Parameters random variables
  gid                   ::    Int         # group ID
  function PMMH(n_iters::Int, smc_alg::SMC, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_iters, smc_alg, space, 0)
  end
end

function Sampler(alg::PMMH)
  info = Dict{Symbol, Any}()
  info[:smc_sampler] = Sampler(SMC(alg.smc_alg, 1))

  # Sanity check for space
  space = union(alg.space, alg.smc_alg.space)
  @assert issubset(Turing._compiler_[:pvars], space) "[Gibbs] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

  if Turing._compiler_[:pvars] != space
  warn("[Gibbs] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
  end

  Sampler(alg, info)
end

step(model::Function, spl::Sampler{PMMH}, vi::VarInfo, is_first::Bool) = begin
  if is_first
    spl.info[:old_like_estimator] = -Inf
    spl.info[:accept_his] = []
    push!(spl.info[:accept_his], true)

    vi
  else
    smc_spl = spl.info[:smc_sampler]
    new_likelihood_estimator = 0.0

    old_θ = copy(vi[spl])
    old_z = copy(vi[smc_spl])

    dprintln(2, "Propose new parameters...")
    vi[getretain(vi, 0, spl)] = NULL
    vi = model(vi=vi, sampler=spl)

    dprintln(2, "Propose new state with SMC...")
    Ws_sum_prev = zeros(Float64, smc_spl.alg.n_particles)
    particles = ParticleContainer{TraceR}(model)

    vi.index = 0; vi.num_produce = 0;  # We need this line cause fork2 deepcopy `vi`.

    vi[getretain(vi, 0, smc_spl)] = NULL
    push!(particles, smc_spl.alg.n_particles, smc_spl, vi)

    while consume(particles) != Val{:done}
      # Compute marginal likehood unbiased estimator
      log_Ws = particles.logWs - Ws_sum_prev # particles.logWs is the running sum over time
      Ws_sum_prev = copy(particles.logWs)
      relative_Ws = exp(log_Ws-maximum(log_Ws))
      logZs = log(sum(relative_Ws)) + maximum(log_Ws)

      new_likelihood_estimator += logZs

      # Resample if needed
      ess = effectiveSampleSize(particles)
      if ess <= smc_spl.alg.resampler_threshold * length(particles)
        resample!(particles,smc_spl.alg.resampler,use_replay=smc_spl.alg.use_replay)
        Ws_sum_prev = 0.0 # Resampling reset weights to zero
      end
    end

    dprintln(2, "computing accept rate α...")
    α = new_likelihood_estimator - spl.info[:old_like_estimator]

    dprintln(2, "decide wether to accept...")
    if log(rand()) < α             # accepted
      ## pick a particle to be retained.
      Ws, _ = weights(particles)
      indx = randcat(Ws)
      vi = particles[indx].vi

      push!(spl.info[:accept_his], true)
      spl.info[:old_like_estimator] = new_likelihood_estimator
    else                      # rejected
      push!(spl.info[:accept_his], false)
      vi[spl] = old_θ
      vi[smc_spl] = old_z
    end

    vi
  end
end

sample(model::Function, alg::PMMH;
       save_state=false,         # flag for state saving
       resume_from=nothing,      # chain to continue
       reuse_spl_n=0             # flag for spl re-using
      ) = begin

    spl = Sampler(alg)
    smc_spl = spl.info[:smc_sampler]

    # Number of samples to store
    sample_n = spl.alg.n_iters

    # Init samples
    time_total = zero(Float64)
    samples = Array{Sample}(sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Init parameters
    vi = resume_from == nothing ?
              model() :
              resume_from.info[:vi]
    n = spl.alg.n_iters

    # PMMH steps
    if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[PMMH] Sampling...", 0) end
    for i = 1:n
      dprintln(2, "PMMH stepping...")
      time_elapsed = @elapsed vi = step(model, spl, vi, i==1)

      if spl.info[:accept_his][end]     # accepted => store the new predcits
        samples[i].value = Sample(vi, spl).value
      else                              # rejected => store the previous predcits
        samples[i] = samples[i - 1]
      end

      time_total += time_elapsed
      if PROGRESS
        haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
      end
    end

    println("[PMMH] Finished with")
    println("  Running time    = $time_total;")

    if resume_from != nothing   # concat samples
      unshift!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    if save_state               # save state
      save!(c, spl, model, vi)
    end

    c
end

assume(spl::Sampler{PMMH}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
    if isempty(spl.alg.space) || vn.sym in spl.alg.space
      vi.index += 1
      if ~haskey(vi, vn)
        r = rand(dist)
        push!(vi, vn, r, dist, spl.alg.gid)
        spl.info[:cache_updated] = CACHERESET # sanity flag mask for getidcs and getranges
        r
      elseif isnan(vi, vn)
        r = rand(dist)
        setval!(vi, vectorize(dist, r), vn)
        setgid!(vi, spl.alg.gid, vn)
        r
      else
        checkindex(vn, vi, spl)
        updategid!(vi, vn, spl)
        vi[vn]
      end
    else
      vi[vn]
    end
end

assume{D<:Distribution}(spl::Sampler{PMMH}, dists::Vector{D}, vn::VarName, var::Any, vi::VarInfo) =
  error("[Turing] PMMH doesn't support vectorizing assume statement")

observe(spl::Sampler{PMMH}, d::Distribution, value::Any, vi::VarInfo) =
  observe(nothing, d, value, vi)

observe{D<:Distribution}(spl::Sampler{PMMH}, ds::Vector{D}, value::Any, vi::VarInfo) =
  observe(nothing, ds, value, vi)
