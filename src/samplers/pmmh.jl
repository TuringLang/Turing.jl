doc"""
    PMMH(n_iters::Int, smc_alg:::SMC,)

Particle marginal Metropolis–Hastings sampler.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), :v2)
alg = PMMH(100, SMC(20, :v1), (:v2, (x) -> Normal(x, 1)))
```
"""
immutable PMMH <: InferenceAlgorithm
  n_iters               ::    Int               # number of iterations
  smc_alg               ::    SMC               # SMC targeting state
  proposals             ::    Dict{Symbol,Any}  # Proposals for paramters
  space                 ::    Set               # Parameters random variables
  gid                   ::    Int               # group ID
  function PMMH(n_iters::Int, smc_alg::SMC, space...)
    new_space = Set()
    proposals = Dict{Symbol,Any}()
    for element in space
        if isa(element, Symbol)
          push!(new_space, element)
        else
          @assert isa(element[1], Symbol) "[PMMH] ($element[1]) should be a Symbol. For proposal, use the syntax PMMH(N, SMC(M, :z), (:m, (x) -> Normal(x, 0.1)))"
          push!(new_space, element[1])
          proposals[element[1]] = element[2]
        end
    end

    new_space = Set(new_space)
    new(n_iters, smc_alg, proposals, new_space, 0)
  end
end

function Sampler(alg::PMMH)
  info = Dict{Symbol, Any}()
  info[:smc_sampler] = Sampler(SMC(alg.smc_alg, 1))

  # Sanity check for space
  space = union(alg.space, alg.smc_alg.space)
  if !isempty(alg.space) || !isempty(alg.smc_alg.space)
    @assert issubset(Turing._compiler_[:pvars], space) "[PMMH] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

    if Turing._compiler_[:pvars] != space
      warn("[PMMH] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
    end
  end

  Sampler(alg, info)
end

step(model::Function, spl::Sampler{PMMH}, vi::VarInfo, is_first::Bool) = begin
  if is_first
    spl.info[:old_likelihood_estimator] = -Inf
    spl.info[:old_prior_prob] = 0.0
    spl.info[:accept_his] = []
  end

  smc_spl = spl.info[:smc_sampler]
  smc_spl.info[:logevidence] = []
  spl.info[:new_prior_prob] = 0.0
  spl.info[:proposal_prob] = 0.0
  spl.info[:violating_support] = false

  old_z = copy(vi[smc_spl])

  dprintln(2, "Propose new parameters from proposals...")
  if !isempty(spl.alg.space)
    old_θ = copy(vi[spl])

    vi = model(vi=vi, sampler=spl)

    if spl.info[:violating_support]
      dprintln(2, "Early rejection, proposal is outside support...")
      push!(spl.info[:accept_his], false)
      vi[spl] = old_θ
      return vi
    end
  end

  dprintln(2, "Propose new state with SMC...")
  vi = step(model, smc_spl, vi)

  dprintln(2, "computing accept rate α...")
  α = smc_spl.info[:logevidence][end] - spl.info[:old_likelihood_estimator]
  if !isempty(spl.alg.proposals)
    α += spl.info[:new_prior_prob] - spl.info[:old_prior_prob] + spl.info[:proposal_prob]
  end

  dprintln(2, "decide wether to accept...")
  if log(rand()) < α             # accepted
    ## pick a particle to be retained.
    push!(spl.info[:accept_his], true)
    spl.info[:old_likelihood_estimator] = smc_spl.info[:logevidence][end]
    spl.info[:old_prior_prob] = spl.info[:new_prior_prob]
  else                      # rejected
    push!(spl.info[:accept_his], false)
    if !isempty(spl.alg.space) vi[spl] = old_θ end
    vi[smc_spl] = old_z
  end

  vi
end

sample(model::Function, alg::PMMH;
       save_state=false,         # flag for state saving
       resume_from=nothing,      # chain to continue
       reuse_spl_n=0             # flag for spl re-using
      ) = begin

    spl = Sampler(alg)

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
        samples[i].value = Sample(vi).value
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
    accept_rate = sum(spl.info[:accept_his]) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
      unshift!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    if save_state               # save state
      save!(c, spl, model, vi)
    end

    c
end

function rand_truncated(dist, lowerbound, upperbound)
    notvalid = true
    x = 0.0
    while (notvalid)
        x = rand(dist)
        notvalid = ((x < lowerbound) | (x > upperbound))
    end
    return x
end

assume(spl::Sampler{PMMH}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
    if isempty(spl.alg.space) || vn.sym in spl.alg.space
      vi.index += 1
      if ~haskey(vi, vn) #NOTE: When would that happens ??
        r = rand(dist)
        push!(vi, vn, r, dist, spl.alg.gid)
        spl.info[:cache_updated] = CACHERESET # sanity flag mask for getidcs and getranges
      elseif vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter
        oldval = getval(vi, vn)[1]
        proposal = spl.alg.proposals[vn.sym](oldval)
        if typeof(proposal) == Distributions.Normal{Float64} # If Gaussian proposal
          σ = std(proposal)
          lb = support(dist).lb
          ub = support(dist).ub
          stdG = Normal()
          r = rand_truncated(proposal, lb, ub)
          # cf http://fsaad.scripts.mit.edu/randomseed/metropolis-hastings-sampling-with-gaussian-drift-proposal-on-bounded-support/
          spl.info[:proposal_prob] += log(cdf(stdG, (ub-oldval)/σ) - cdf(stdG,(lb-oldval)/σ))
          spl.info[:proposal_prob] -= log(cdf(stdG, (ub-r)/σ) - cdf(stdG,(lb-r)/σ))
      else # Other than Gaussian proposal
          r = rand(proposal)
          if (r < support(dist).lb) | (r > support(dist).ub) # check if value lies in support
            spl.info[:violating_support] = true
            r = oldval
          end
          spl.info[:proposal_prob] -= logpdf(proposal, r) # accumulate pdf of proposal
          reverse_proposal = spl.alg.proposals[vn.sym](r)
          spl.info[:proposal_prob] += logpdf(reverse_proposal, oldval)
        end
        spl.info[:new_prior_prob] += logpdf(dist, r) # accumulate pdf of prior
      else # Prior as proposal
        r = rand(dist)
      end
      setval!(vi, vectorize(dist, r), vn)
      setgid!(vi, spl.alg.gid, vn)
      r
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
