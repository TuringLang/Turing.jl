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
  m ~ Normal(0,sqrt.(s))
  x[1] ~ Normal(m, sqrt.(s))
  x[2] ~ Normal(m, sqrt.(s))
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
    n_adapt_default = Int(round(n_iters / 2))
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

      θ = realpart(vi[spl])
      ϵ = spl.alg.delta > 0 ?
          find_good_eps(model, vi, spl) :       # heuristically find optimal ϵ
          spl.info[:pre_set_ϵ]
      vi[spl] = θ

      if spl.alg.gid != 0 invlink!(vi, spl) end # R -> X

      push!(spl.info[:wum][:ϵ], ϵ)
      update_da_μ(spl.info[:wum], ϵ)
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

    lj_func(theta) = begin

        vi[spl][:] = theta[:]
        realpart(runmodel(model, vi, spl).logp)

    end

    grad_func(θ::T) where {T<:Union{Vector,SubArray}} = begin

      if ADBACKEND == :forward_diff
        vi[spl] = θ
        grad = gradient(vi, model, spl)
      elseif ADBACKEND == :reverse_diff
        grad = gradient_r(θ, vi, model, spl)
      end
  
      return getlogp(vi), grad
  
    end

    rev_func(θ_old::T, old_logp::R) where {T<:Union{Vector,SubArray},R<:Real} = begin

      if ADBACKEND == :forward_diff
        vi[spl] = θ_old
      elseif ADBACKEND == :reverse_diff
        vi_spl = vi[spl]
        for i = 1:length(θ_old)
          if isa(vi_spl[i], ReverseDiff.TrackedReal)
            vi_spl[i].value = θ_old[i]
          else
            vi_spl[i] = θ_old[i]
          end
        end
      end
      setlogp!(vi, old_logp)
  
    end
  
    log_func() = begin
  
      spl.info[:lf_num] += 1
      spl.info[:total_lf_num] += 1  # record leapfrog num
  
    end

    θ = realpart(vi[spl])
    lj = vi.logp
    stds = spl.info[:wum][:stds]

    θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(θ, lj, lj_func, grad_func, ϵ, λ, stds; 
                                             rev_func=rev_func, log_func=log_func)

    if PROGRESS && spl.alg.gid == 0
      stds_str = string(spl.info[:wum][:stds])
      stds_str = length(stds_str) >= 32 ? stds_str[1:30]*"..." : stds_str
      haskey(spl.info, :progress) && ProgressMeter.update!(
                                       spl.info[:progress],
                                       spl.info[:progress].counter; showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, stds_str)])
    end

    dprintln(2, "decide wether to accept...")
    if is_accept              # accepted
      push!(spl.info[:accept_his], true)

      vi[spl][:] = θ_new[:]
      setlogp!(vi, lj_func(θ_new))
    else                      # rejected
      push!(spl.info[:accept_his], false)

      # Reset Θ
      # NOTE: ForwardDiff and ReverseDiff need different implementation
      #       due to immutable Dual vs mutable TrackedReal
      if ADBACKEND == :forward_diff

        vi[spl] = θ

      elseif ADBACKEND == :reverse_diff

        vi_spl = vi[spl]
        for i = 1:length(θ)
          if isa(vi_spl[i], ReverseDiff.TrackedReal)
            vi_spl[i].value = θ[i]
          else
            vi_spl[i] = θ[i]
          end
        end

      end

      setlogp!(vi, lj)  # reset logp
    end

    # QUES: why do we need the 2nd condition here (Kai)
    if spl.alg.delta > 0 && τ_valid > 0    # only do adaption for HMCDA
      # TODO: figure out why realpart() is needed for α in HMCDA
      adapt!(spl.info[:wum], realpart(α), realpart(vi[spl]), adapt_ϵ = true)
    end

    dprintln(3, "R -> X...")
    if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

    vi
  end
end
