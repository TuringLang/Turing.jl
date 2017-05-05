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
immutable HMCDA <: InferenceAlgorithm
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
    vi_0 = deepcopy(vi)

    if spl.alg.delta > 0

      dprintln(3, "X -> R...")
      if spl.alg.gid != 0 vi = link(vi, spl) end

      # Heuristically find optimal ϵ
      ϵ = find_good_eps(model, spl, vi)

    else
      ϵ = spl.info[:ϵ][end]
    end

    spl.info[:ϵ] = [ϵ]
    spl.info[:μ] = log(10 * ϵ)
    spl.info[:ϵ_bar] = 1.0
    spl.info[:H_bar] = 0.0
    spl.info[:m] = 0

    true, vi_0
  else
    # Set parameters
    δ = spl.alg.delta
    λ = spl.alg.lambda
    ϵ = spl.info[:ϵ][end]

    dprintln(2, "current ϵ: $ϵ")
    μ, γ, t_0, κ = spl.info[:μ], 0.05, 10, 0.75
    ϵ_bar, H_bar = spl.info[:ϵ_bar], spl.info[:H_bar]

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(3, "X -> R...")
    if spl.alg.gid != 0 vi = link(vi, spl) end

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, vi, spl)

    τ = max(1, round(Int, λ / ϵ))
    dprintln(2, "leapfrog for $τ steps with step size $ϵ")
    vi, p, τ_valid = leapfrog(vi, p, τ, ϵ, model, spl)

    dprintln(2, "computing new H...")
    H = τ_valid == 0 ? Inf : find_H(p, model, vi, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    α = min(1, exp(-ΔH))  # MH accept rate

    haskey(spl.info, :progress) && ProgressMeter.update!(
                                     spl.info[:progress],
                                     spl.info[:progress].counter; showvalues = [(:ϵ, ϵ), (:α, α)]
                                   )

    # Use Dual Averaging to adapt ϵ
    m = spl.info[:m] += 1
    if m < spl.alg.n_adapt
      dprintln(1, " ϵ = $ϵ, α = $α, exp(-ΔH)=$(exp(-ΔH))")
      H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - α)
      ϵ = exp(μ - sqrt(m) / γ * H_bar)
      ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
      push!(spl.info[:ϵ], ϵ)
      spl.info[:ϵ_bar], spl.info[:H_bar] = ϵ_bar, H_bar
    elseif m == spl.alg.n_adapt
      dprintln(0, " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption.")
      push!(spl.info[:ϵ], spl.info[:ϵ_bar])
    end

    dprintln(3, "R -> X...")
    if spl.alg.gid != 0 vi = invlink(vi, spl); cleandual!(vi) end

    dprintln(2, "decide wether to accept...")
    if rand() < α      # accepted
      true, vi
    else                                # rejected
      false, vi
    end
  end
end
