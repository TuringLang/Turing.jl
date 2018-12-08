"""
    MH(n_iters::Int)

Metropolis-Hastings sampler.

Usage:

```julia
MH(100, (:m, (x) -> Normal(x, 0.1)))
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

chn = sample(gdemo([1.5, 2]), MH(1000))
```
"""
mutable struct MH{T} <: InferenceAlgorithm
  n_iters   ::  Int       # number of iterations
  proposals ::  Dict{Symbol,Any}  # Proposals for paramters
  space     ::  Set{T}    # sampling space, emtpy means all
  gid       ::  Int       # group ID
end
function MH(n_iters::Int, space...)
  new_space = Set()
  proposals = Dict{Symbol,Any}()

  # parse random variables with their hypothetical proposal
  for element in space
      if isa(element, Symbol)
        push!(new_space, element)
      else
        @assert isa(element[1], Symbol) "[MH] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
        push!(new_space, element[1])
        proposals[element[1]] = element[2]
      end
  end
  set = Set(new_space)
  MH{eltype(set)}(n_iters, proposals, set, 0)
end
MH{T}(alg::MH, new_gid::Int) where T = MH{T}(alg.n_iters, alg.proposals, alg.space, new_gid)

Sampler(model::CallableModel, alg::MH) = begin
  alg_str = "MH"

  # Sanity check for space
  if alg.gid == 0 && !isempty(alg.space)
    @assert issubset(model.pvars, alg.space) "[$alg_str] symbols specified to samplers ($alg.space) doesn't cover the model parameters ($(model.pvars))"
    if model.pvars != alg.space
      warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(setdiff(alg.space, model.pvars))")
    end
  end

  info = Dict{Symbol, Any}()
  info[:proposal_ratio] = 0.0
  info[:prior_prob] = 0.0
  info[:violating_support] = false

  Sampler(alg, info)
end

propose(model, spl::Sampler{<:MH}, vi::VarInfo) = begin
  spl.info[:proposal_ratio] = 0.0
  spl.info[:prior_prob] = 0.0
  spl.info[:violating_support] = false
  runmodel!(model, vi ,spl)
end

function step(model, spl::Sampler{<:MH}, vi::VarInfo, is_first::Val{true})
  return vi, true
end

function step(model, spl::Sampler{<:MH}, vi::VarInfo, is_first::Val{false})
  if spl.alg.gid != 0 # Recompute joint in logp
    runmodel!(model, vi, nothing)
  end
  old_θ = copy(vi[spl])
  old_logp = getlogp(vi)

  @debug "Propose new parameters from proposals..."
  propose(model, spl, vi)

  @debug "computing accept rate α..."
  is_accept, logα = mh_accept(-old_logp, -getlogp(vi), spl.info[:proposal_ratio])

  @debug "decide wether to accept..."
  if is_accept && !spl.info[:violating_support]  # accepted
    is_accept = true
  else                      # rejected
    is_accept = false
    vi[spl] = old_θ         # reset Θ
    setlogp!(vi, old_logp)  # reset logp
  end

  return vi, is_accept
end

function sample(model, alg::MH;
                save_state=false,         # flag for state saving
                resume_from=nothing,      # chain to continue
                reuse_spl_n=0,            # flag for spl re-using
                )

  spl = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        Sampler(alg)
  alg_str = "MH"

  # Initialization
  time_total = 0.0
  n = reuse_spl_n > 0 ?
      reuse_spl_n :
      alg.n_iters
  samples = Array{Sample}(undef, n)
  weight = 1 / n
  for i = 1:n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

    vi = if resume_from == nothing
        vi_ = VarInfo()
        model(vi_, HamiltonianRobustInit())
        vi_
    else
        resume_from.info[:vi]
    end

  if spl.alg.gid == 0
    runmodel!(model, vi, spl)
  end

  # MH steps
  accept_his = Bool[]
  PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
  for i = 1:n
    @debug "$alg_str stepping..."

    time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val(i == 1))
    time_total += time_elapsed

    if is_accept # accepted => store the new predcits
        samples[i].value = Sample(vi, spl).value
    else         # rejected => store the previous predcits
        samples[i] = samples[i - 1]
    end

    samples[i].value[:elapsed] = time_elapsed
    push!(accept_his, is_accept)

    PROGRESS[] && (ProgressMeter.next!(spl.info[:progress]))
  end

  println("[$alg_str] Finished with")
  println("  Running time        = $time_total;")
  accept_rate = sum(accept_his) / n  # calculate the accept rate
  println("  Accept rate         = $accept_rate;")

  if resume_from != nothing   # concat samples
    pushfirst!(samples, resume_from.value2...)
  end
  c = Chain(0, samples)       # wrap the result by Chain
  if save_state               # save state
    save!(c, spl, model, vi)
  end

  c
end

assume(spl::Sampler{<:MH}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
    if isempty(spl.alg.space) || vn.sym in spl.alg.space
      if ~haskey(vi, vn) error("[MH] does not handle stochastic existence yet") end
      old_val = vi[vn]

      if vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter
        proposal = spl.alg.proposals[vn.sym](old_val)

        if typeof(proposal) == Distributions.Normal{Float64} # If Gaussian proposal
          σ = std(proposal)
          lb = support(dist).lb
          ub = support(dist).ub
          stdG = Normal()
          r = rand(TruncatedNormal(proposal.μ, proposal.σ, lb, ub))
          # cf http://fsaad.scripts.mit.edu/randomseed/metropolis-hastings-sampling-with-gaussian-drift-proposal-on-bounded-support/
          spl.info[:proposal_ratio] += log(cdf(stdG, (ub-old_val)/σ) - cdf(stdG,(lb-old_val)/σ))
          spl.info[:proposal_ratio] -= log(cdf(stdG, (ub-r)/σ) - cdf(stdG,(lb-r)/σ))

        else # Other than Gaussian proposal
          r = rand(proposal)
          if (r < support(dist).lb) | (r > support(dist).ub) # check if value lies in support
            spl.info[:violating_support] = true
            r = old_val
          end
          spl.info[:proposal_ratio] -= logpdf(proposal, r) # accumulate pdf of proposal
          reverse_proposal = spl.alg.proposals[vn.sym](r)
          spl.info[:proposal_ratio] += logpdf(reverse_proposal, old_val)
        end

      else # Prior as proposal
        r = rand(dist)
        spl.info[:proposal_ratio] += (logpdf(dist, old_val) - logpdf(dist, r))
      end

      spl.info[:prior_prob] += logpdf(dist, r) # accumulate prior for PMMH
      vi[vn] = vectorize(dist, r)
      setgid!(vi, spl.alg.gid, vn)
    else
      r = vi[vn]
    end

    # acclogp!(vi, logpdf(dist, r)) # accumulate pdf of prior
    r, logpdf(dist, r)
end

assume(spl::Sampler{<:MH}, dists::Vector{D}, vn::VarName, var::Any, vi::VarInfo) where D<:Distribution =
  error("[Turing] MH doesn't support vectorizing assume statement")

observe(spl::Sampler{<:MH}, d::Distribution, value::Any, vi::VarInfo) =
  observe(nothing, d, value, vi)  # accumulate pdf of likelihood

observe(spl::Sampler{<:MH}, ds::Vector{D}, value::Any, vi::VarInfo)  where D<:Distribution =
  observe(nothing, ds, value, vi) # accumulate pdf of likelihood
