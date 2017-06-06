doc"""
    HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64)

Hamiltonian Monte Carlo sampler wiht Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
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

sample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))
```
"""
immutable HMCDA <: Hamiltonian
  n_iters   ::  Int       # number of samples
  n_adapt   ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  lambda    ::  Float64   # target leapfrog length
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int       # group ID

  HMCDA(n_adapt::Int, delta::Float64, lambda::Float64, space...) = new(1, n_adapt, delta, lambda, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  HMCDA(n_iters::Int, delta::Float64, lambda::Float64) = begin
    n_adapt_default = Int(round(n_iters / 5))
    new(n_iters, n_adapt_default > 1000 ? 1000 : n_adapt_default, delta, lambda, Set(), 0)
  end
  HMCDA(alg::HMCDA, new_gid::Int) =
    new(alg.n_iters, alg.n_adapt, alg.delta, alg.lambda, alg.space, new_gid)
  HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64) =
    new(n_iters, n_adapt, delta, lambda, Set(), 0)
  HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64, space...) =
    new(n_iters, n_adapt, delta, lambda, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64, space::Set, gid::Int) =
    new(n_iters, n_adapt, delta, lambda, space, gid)
end

function step(model, spl::Sampler{HMCDA}, vi::VarInfo, is_first::Bool)
  if is_first
    if ~haskey(spl.info, :wum)
      if spl.alg.gid != 0 link!(vi, spl) end    # X -> R

      init_warm_up_params(vi, spl)

      ϵ = spl.alg.delta > 0 ?
          find_good_eps(model, vi, spl) :       # heuristically find optimal ϵ
          spl.info[:pre_set_ϵ]

      if spl.alg.gid != 0 invlink!(vi, spl) end # R -> X

      update_da_params(spl.info[:wum], ϵ)
    end

    push!(spl.info[:accept_his], true)

    vi
  else
    # Set parameters
    λ = spl.alg.lambda
    ϵ = spl.info[:wum][:ϵ][end]; dprintln(2, "current ϵ: $ϵ")

    spl.info[:lf_num] = 0   # reset current lf num counter

    dprintln(3, "X-> R...")
    if spl.alg.gid != 0
      link!(vi, spl)
      runmodel(model, vi, spl)
    end

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(2, "recording old values...")
    old_θ = vi[spl]; old_logp = getlogp(vi)
    old_H = find_H(p, model, vi, spl)

    τ = max(1, round(Int, λ / ϵ))
    dprintln(2, "leapfrog for $τ steps with step size $ϵ")
    θ, p, τ_valid = leapfrog2(old_θ, p, τ, ϵ, model, vi, spl)

    dprintln(2, "computing new H...")
    H = τ_valid == 0 ? Inf : find_H(p, model, vi, spl)

    dprintln(2, "computing accept rate α...")
    α = min(1, exp(-(H - old_H)))

    if ~(isdefined(Main, :IJulia) && Main.IJulia.inited) # Fix for Jupyter notebook.
    stds_str = string(spl.info[:wum][:stds])
    stds_str = length(stds_str) >= 32 ? stds_str[1:30]*"..." : stds_str
    haskey(spl.info, :progress) && ProgressMeter.update!(
                                     spl.info[:progress],
                                     spl.info[:progress].counter; showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, stds_str)]
                                   )
    end

    dprintln(2, "decide wether to accept...")
    if rand() < α             # accepted
      push!(spl.info[:accept_his], true)
    else                      # rejected
      push!(spl.info[:accept_his], false)
      vi[spl] = old_θ         # reset Θ
      setlogp!(vi, old_logp)  # reset logp
    end

    # Adapt step-size and pre-cond
    adapt(spl.info[:wum], α, realpart(vi[spl]))

    dprintln(3, "R -> X...")
    if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

    vi
  end
end
