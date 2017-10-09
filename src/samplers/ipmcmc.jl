doc"""
    IPMCMC(n_particles::Int, n_iters::Int)

Particle Gibbs sampler.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), IPMCMC(100, 100, 4, 2))
```
"""
immutable IPMCMC <: InferenceAlgorithm
  n_particles           ::    Int         # number of particles used
  n_iters               ::    Int         # number of iterations
  n_nodes               ::    Int         # number of nodes running SMC and CSMC
  n_csmc_nodes          ::    Int         # number of nodes CSMC
  resampler             ::    Function    # function to resample
  resampler_threshold   ::    Float64     # threshold of ESS for resampling
  space                 ::    Set         # sampling space, emtpy means all
  gid                   ::    Int         # group ID
  IPMCMC(n1::Int, n2::Int) = new(n1, n2, 32, 16, resampleSystematic, 0.5, Set(), 0)
  IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int) = new(n1, n2, n3, n4, resampleSystematic, 0.5, Set(), 0)
  function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n1, n2, n3, n4, resampleSystematic, 0.5, space, 0)
  end
  IPMCMC(alg::IPMCMC, new_gid::Int) = new(alg.n_particles, alg.n_iters, alg.n_nodes, alg.n_csmc_nodes, alg.resampler, alg.resampler_threshold, alg.space, new_gid)
end

function Sampler(alg::IPMCMC)
  samplers = Array{Sampler}(alg.n_nodes)
  default_CSMC = CSMC(alg.n_particles, 1, alg.resampler, alg.resampler_threshold, alg.space, 0)
  default_SMC = SMC(alg.n_particles, alg.resampler, alg.resampler_threshold, true, alg.space, 0)

  for i in 1:alg.n_csmc_nodes
    samplers[i] = Sampler(CSMC(default_CSMC, i))
  end
  for i in (alg.n_csmc_nodes+1):alg.n_nodes
    samplers[i] = Sampler(SMC(default_SMC, i))
  end

  info = Dict{Symbol, Any}()
  info[:samplers] = samplers

  Sampler(alg, info)
end

step(model::Function, spl::Sampler{SMC}, vi::VarInfo) = begin

    Ws_sum_prev = zeros(Float64, spl.alg.n_particles)
    likelihood_estimator = 0.0

    # NOTE: Should use replay ??
    TraceType = spl.alg.use_replay ? TraceR : TraceC
    particles = ParticleContainer{TraceType}(model)
    vi.index = 0; vi.num_produce = 0;  # We need this line cause fork2 deepcopy `vi`.
    vi[getretain(vi, 0, spl)] = NULL
    push!(particles, spl.alg.n_particles, spl, vi)

    while consume(particles) != Val{:done}
      # Compute marginal likehood unbiased estimator
      log_Ws = particles.logWs - Ws_sum_prev # particles.logWs is the running sum over time
      Ws_sum_prev = copy(particles.logWs)
      relative_Ws = exp(log_Ws-maximum(log_Ws))
      logZs = log(sum(relative_Ws)) + maximum(log_Ws)
      likelihood_estimator += logZs

      # Resample if needed
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler,use_replay=spl.alg.use_replay)
        Ws_sum_prev = zeros(Float64, spl.alg.n_particles) # Resampling reset weights to zero
      end
    end

    spl.info[:particles] = particles
    setlogp!(vi, likelihood_estimator)

    vi
end

step(model::Function, spl::Sampler{IPMCMC}, vi::VarInfo, is_first::Bool) = begin
  if is_first
    VarInfos = Array{VarInfo}(spl.alg.n_nodes)
    for j in 1:spl.alg.n_nodes
      VarInfos[j] = model()
    end
  else
    VarInfos = spl.info[:VarInfos]
  end
  log_zs = zeros(spl.alg.n_nodes)

  if spl.alg.gid != 0
    for j in 1:spl.alg.n_csmc_nodes
      value = copy(VarInfos[j][spl])
      VarInfos[j] = deepcopy(vi)
      VarInfos[j][spl] = value
    end
    for j in spl.alg.n_csmc_nodes+1:spl.alg.n_nodes
      VarInfos[j] = deepcopy(vi)
    end
  end

  # SMC & CSMC workers
  for j in 1:spl.alg.n_nodes
    VarInfos[j] = step(model, spl.info[:samplers][j], VarInfos[j])
    log_zs[j] = getlogp(VarInfos[j])
  end

  # Resampling of CSMC nodes indices
  for j in 1:spl.alg.n_csmc_nodes
    # Select a new conditional node by simulating cj
    ksi = vcat(log_zs[spl.alg.n_csmc_nodes+1:spl.alg.n_nodes], log_zs[j])

    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    gs = -log(-log(rand(length(ksi))))
    _, selected_index = findmax(gs + ksi)
    if selected_index < length(ksi)
      particles = spl.info[:samplers][spl.alg.n_csmc_nodes+selected_index].info[:particles]
      Ws, _ = weights(particles)
      indx = randcat(Ws)
      VarInfos[j] = deepcopy(particles[indx].vi)
    end
  end

  spl.info[:VarInfos] = VarInfos

  VarInfos[1]
end

sample(model::Function, alg::IPMCMC) = begin

  spl = Sampler(alg)

  # Number of samples to store
  sample_n = alg.n_iters * alg.n_csmc_nodes

  # Init samples
  time_total = zero(Float64)
  samples = Array{Sample}(alg.n_iters, alg.n_csmc_nodes)
  weight = 1 / sample_n
  for i = 1:alg.n_iters
    for j in 1:spl.alg.n_csmc_nodes
      samples[i,j] = Sample(weight, Dict{Symbol, Any}())
    end
  end

  # Init parameters
  vi = model()
  n = spl.alg.n_iters

  # IPMCMC steps
  if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[IPMCMC] Sampling...", 0) end
  for i = 1:n
    dprintln(2, "IPMCMC stepping...")
    time_elapsed = @elapsed vi = step(model, spl, vi, i==1)

    for j in 1:spl.alg.n_csmc_nodes
      samples[i,j].value = Sample(spl.info[:VarInfos][j], spl).value
    end

    time_total += time_elapsed
    if PROGRESS
      haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
    end
  end

  println("[IPMCMC] Finished with")
  println("  Running time    = $time_total;")

  Chain(0, reshape(transpose(samples), sample_n)) # wrap the result by Chain
end
