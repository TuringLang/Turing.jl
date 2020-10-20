
####
#### Particle marginal Metropolis-Hastings sampler.
####

"""
    PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})

Particle independant Metropolis–Hastings and
Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

- `n_iters::Int` : Number of iterations to run.
- `smc_alg:::SMC` : An [`SMC`](@ref) algorithm to use.
- `parameters_algs::Tuple{MH}` : An [`MH`](@ref) algorithm, which includes a
sample space specification.
"""
mutable struct PMMH{space, A<:Tuple} <: InferenceAlgorithm
    n_iters::Int               # number of iterations
    algs::A                 # Proposals for state & parameters
end
function PMMH(n_iters::Int, algs::Tuple, space::Tuple)
    return PMMH{space, typeof(algs)}(n_iters, algs)
end
function PMMH(n_iters::Int, smc_alg::SMC, parameter_algs...)
    return PMMH(n_iters, tuple(parameter_algs..., smc_alg), ())
end

PIMH(n_iters::Int, smc_alg::SMC) = PMMH(n_iters, tuple(smc_alg), ())

function Sampler(alg::PMMH, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    spl = Sampler(alg, info, s)

    alg_str = "PMMH"
    n_samplers = length(alg.algs)
    samplers = Array{Sampler}(undef, n_samplers)

    space = Set{Symbol}()

    for i in 1:n_samplers
        sub_alg = alg.algs[i]
        if isa(sub_alg, Union{SMC, MH})
            samplers[i] = Sampler(sub_alg, model, Selector(Symbol(typeof(sub_alg))))
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
        if typeof(sub_alg) == MH && sub_alg.n_iters != 1
            warn(
                "[$alg_str] number of iterations greater than 1" * 
                "is useless for MH since it is only used for its proposal"
            )
        end
        space = union(space, sub_alg.space)
    end

    info[:old_likelihood_estimate] = -Inf # Force to accept first proposal
    info[:old_prior_prob] = 0.0
    info[:samplers] = samplers

    return spl
end

function step(model, spl::Sampler{<:PMMH}, vi::VarInfo, is_first::Bool)
    violating_support = false
    proposal_ratio = 0.0
    new_prior_prob = 0.0
    new_likelihood_estimate = 0.0
    old_θ = copy(vi[spl])

    @debug "Propose new parameters from proposals..."
    for local_spl in spl.info[:samplers][1:end-1]
        @debug "$(typeof(local_spl)) proposing $(local_spl.alg.space)..."
        propose(model, local_spl, vi)
        if local_spl.info[:violating_support] violating_support=true; break end
        new_prior_prob += local_spl.info[:prior_prob]
        proposal_ratio += local_spl.info[:proposal_ratio]
    end

    if violating_support
        # do not run SMC if going to refuse anyway
        accepted = false
    else
        @debug "Propose new state with SMC..."
        vi, _ = step(model, spl.info[:samplers][end], vi)
        new_likelihood_estimate = spl.info[:samplers][end].info[:logevidence][end]

        @debug "Decide whether to accept..."
        accepted = mh_accept(
          spl.info[:old_likelihood_estimate] + spl.info[:old_prior_prob],
          new_likelihood_estimate + new_prior_prob,
          proposal_ratio,
        )
    end

    if accepted
        spl.info[:old_likelihood_estimate] = new_likelihood_estimate
        spl.info[:old_prior_prob] = new_prior_prob
    else                      # rejected
        vi[spl] = old_θ
    end

    return vi, accepted
end

function sample(  model::Model,
                  alg::PMMH;
                  save_state=false,         # flag for state saving
                  resume_from=nothing,      # chain to continue
                  reuse_spl_n=0             # flag for spl re-using
                )

    spl = Sampler(alg, model)
    if resume_from !== nothing
        spl.selector = resume_from.info[:spl].selector
    end
    alg_str = "PMMH"

    # Number of samples to store
    sample_n = spl.alg.n_iters

    # Init samples
    time_total = zero(Float64)
    samples = Array{Sample}(undef, sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Init parameters
    vi = if resume_from === nothing
        vi_ = VarInfo(model)
    else
        resume_from.info[:vi]
    end
    n = spl.alg.n_iters

    # PMMH steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
      @debug "$alg_str stepping..."
      time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, i==1)

      if is_accept # accepted => store the new predcits
          samples[i].value = Sample(vi, spl).value
      else         # rejected => store the previous predcits
          samples[i] = samples[i - 1]
      end

      time_total += time_elapsed
      push!(accept_his, is_accept)
      if PROGRESS[]
        haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
      end
    end

    println("[$alg_str] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from !== nothing   # concat samples
      pushfirst!(samples, resume_from.info[:samples]...)
    end
    c = Chain(-Inf, samples)       # wrap the result by Chain

    if save_state               # save state
      c = save(c, spl, model, vi, samples)
    end

    c
end


####
#### IMCMC Sampler.
####

"""
    IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

- `n_particles::Int` : Number of particles to use.
- `n_iters::Int` : Number of iterations to employ.
- `n_nodes::Int` : The number of nodes running SMC and CSMC.
- `n_csmc_nodes::Int` : The number of CSMC nodes.
```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).
"""
mutable struct IPMCMC{T, F} <: InferenceAlgorithm
  n_particles::Int         # number of particles used
  n_iters::Int         # number of iterations
  n_nodes::Int         # number of nodes running SMC and CSMC
  n_csmc_nodes::Int         # number of nodes CSMC
  resampler::F           # function to resample
  space::Set{T}      # sampling space, emtpy means all
end
IPMCMC(n1::Int, n2::Int) = IPMCMC(n1, n2, 32, 16, resample_systematic, Set())
IPMCMC(n1::Int, n2::Int, n3::Int) = IPMCMC(n1, n2, n3, Int(ceil(n3/2)), resample_systematic, Set())
IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int) = IPMCMC(n1, n2, n3, n4, resample_systematic, Set())
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int, space...)
  _space = isa(space, Symbol) ? Set([space]) : Set(space)
  IPMCMC(n1, n2, n3, n4, resample_systematic, _space)
end

function Sampler(alg::IPMCMC, s::Selector)
  info = Dict{Symbol, Any}()
  spl = Sampler(alg, info, s)
  # Create SMC and CSMC nodes
  samplers = Array{Sampler}(undef, alg.n_nodes)
  # Use resampler_threshold=1.0 for SMC since adaptive resampling is invalid in this setting
  default_CSMC = CSMC(alg.n_particles, 1, alg.resampler, alg.space)
  default_SMC = SMC(alg.n_particles, alg.resampler, 1.0, false, alg.space)

  for i in 1:alg.n_csmc_nodes
    samplers[i] = Sampler(default_CSMC, Selector(Symbol(typeof(default_CSMC))))
  end
  for i in (alg.n_csmc_nodes+1):alg.n_nodes
    samplers[i] = Sampler(default_SMC, Selector(Symbol(typeof(default_CSMC))))
  end

  info[:samplers] = samplers

  return spl
end

function step(model, spl::Sampler{<:IPMCMC}, VarInfos::Array{VarInfo}, is_first::Bool)
    # Initialise array for marginal likelihood estimators
    log_zs = zeros(spl.alg.n_nodes)

    # Run SMC & CSMC nodes
    for j in 1:spl.alg.n_nodes
        reset_num_produce!(VarInfos[j])
        VarInfos[j] = step(model, spl.info[:samplers][j], VarInfos[j])[1]
        log_zs[j] = spl.info[:samplers][j].info[:logevidence][end]
    end

    # Resampling of CSMC nodes indices
    conditonal_nodes_indices = collect(1:spl.alg.n_csmc_nodes)
    unconditonal_nodes_indices = collect(spl.alg.n_csmc_nodes+1:spl.alg.n_nodes)
    for j in 1:spl.alg.n_csmc_nodes
        # Select a new conditional node by simulating cj
        log_ksi = vcat(log_zs[unconditonal_nodes_indices], log_zs[j])
        ksi = exp.(log_ksi .- maximum(log_ksi))
        c_j = wsample(ksi) # sample from Categorical with unormalized weights

        if c_j < length(log_ksi) # if CSMC node selects another index than itself
            conditonal_nodes_indices[j] = unconditonal_nodes_indices[c_j]
            unconditonal_nodes_indices[c_j] = j
        end
    end
    nodes_permutation = vcat(conditonal_nodes_indices, unconditonal_nodes_indices)

    VarInfos[nodes_permutation]
end

function sample(model::Model, alg::IPMCMC)

  spl = Sampler(alg)

  # Number of samples to store
  sample_n = alg.n_iters * alg.n_csmc_nodes

  # Init samples
  time_total = zero(Float64)
  samples = Array{Sample}(undef, sample_n)
  weight = 1 / sample_n
  for i = 1:sample_n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

  # Init parameters
  vi = empty!(VarInfo(model))
  VarInfos = Array{VarInfo}(undef, spl.alg.n_nodes)
  for j in 1:spl.alg.n_nodes
    VarInfos[j] = deepcopy(vi)
  end
  n = spl.alg.n_iters

  # IPMCMC steps
  if PROGRESS[] spl.info[:progress] = ProgressMeter.Progress(n, 1, "[IPMCMC] Sampling...", 0) end
  for i = 1:n
    @debug "IPMCMC stepping..."
    time_elapsed = @elapsed VarInfos = step(model, spl, VarInfos, i==1)

    # Save each CSMS retained path as a sample
    for j in 1:spl.alg.n_csmc_nodes
      samples[(i-1)*alg.n_csmc_nodes+j].value = Sample(VarInfos[j], spl).value
    end

    time_total += time_elapsed
    if PROGRESS[]
      haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
    end
  end

  println("[IPMCMC] Finished with")
  println("  Running time    = $time_total;")

  Chain(0.0, samples) # wrap the result by Chain
end
