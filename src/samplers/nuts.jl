doc"""
    NUTS(n_iters::Int, n_adapt::Int, delta::Float64)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.65)
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

sample(gdemo([1.5, 2]), NUTS(1000, 200, 0.65))
```
"""
immutable NUTS <: InferenceAlgorithm
  n_iters   ::  Int       # number of samples
  n_adapt   ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int       # group ID

  NUTS(n_adapt::Int, delta::Float64, space...) =
    new(1, n_adapt, delta, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  NUTS(n_iters::Int, n_adapt::Int, delta::Float64, space...) =
    new(n_iters, n_adapt, delta, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  NUTS(n_iters::Int, delta::Float64) = begin
    n_adapt_default = Int(round(n_iters / 5))
    new(n_iters, n_adapt_default > 1000 ? 1000 : n_adapt_default, delta, Set(), 0)
  end
  NUTS(alg::NUTS, new_gid::Int) = new(alg.n_iters, alg.n_adapt, alg.delta, alg.space, new_gid)
end

function step(model::Function, spl::Sampler{NUTS}, vi::VarInfo, is_first::Bool)
  if is_first

    spl.info[:stds] = ones(length(vi[spl]))

    if spl.alg.gid != 0 link!(vi, spl) end      # X -> R

    spl.info[:θ_mean] = realpart(vi[spl])
    spl.info[:θ_num] = 1
    D = length(vi[spl])
    spl.info[:stds] = ones(D)
    spl.info[:θ_vars] = nothing

    ϵ = find_good_eps(model, vi, spl)           # heuristically find optimal ϵ

    if spl.alg.gid != 0 invlink!(vi, spl) end   # R -> X

    spl.info[:ϵ] = ϵ
    spl.info[:μ] = log(10 * ϵ)
    spl.info[:ϵ_bar] = 1.0
    # spl.info[:ϵ_bar] = ϵ_bar  # NOTE: is this correct?
    spl.info[:H_bar] = 0.0
    spl.info[:m] = 0

    push!(spl.info[:accept_his], true)

    vi
  else
    # Set parameters
    δ = spl.alg.delta
    ϵ = spl.info[:ϵ]

    dprintln(2, "current ϵ: $ϵ")
    μ, γ, t_0, κ = spl.info[:μ], 0.05, 10, 0.75
    ϵ_bar, H_bar = spl.info[:ϵ_bar], spl.info[:H_bar]

    dprintln(3, "X -> R...")
    if spl.alg.gid != 0
      link!(vi, spl)
      runmodel(model, vi, spl)
    end

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(2, "recording old H...")
    H0 = find_H(p, model, vi, spl)

    dprintln(3, "sample slice variable u")
    logu = log(rand()) + (-H0)

    θ = vi[spl]
    logp = getlogp(vi)
    θm, θp, rm, rp, j, n, s = θ, θ, p, p, 0, 1, 1

    local α, n_α
    while s == 1 && j <= 5
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, θ′, logp′, n′, s′, α, n_α = build_tree(θm, rm, logu, v_j, j, ϵ, H0, model, spl, vi)
      else
        _, _, θp, rp, θ′, logp′, n′, s′, α, n_α = build_tree(θp, rp, logu, v_j, j, ϵ, H0, model, spl, vi)
      end

      if ~(isdefined(Main, :IJulia) && Main.IJulia.inited) # Fix for Jupyter notebook.
      haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress],
                                  spl.info[:progress].counter;
                                  showvalues = [(:ϵ, ϵ), (:tree_depth, j)])
      end

      if s′ == 1 && rand() < min(1, n′ / n)
        θ = θ′
        logp = logp′
      end
      n = n + n′

      s = s′ * (dot(θp - θm, rm) >= 0 ? 1 : 0) * (dot(θp - θm, rp) >= 0 ? 1 : 0)
      j = j + 1
    end

    # Use Dual Averaging to adapt ϵ
    m = spl.info[:m] += 1
    if m < spl.alg.n_adapt
      H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - α / n_α)
      ϵ = exp(μ - sqrt(m) / γ * H_bar)
      ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
      spl.info[:ϵ] = ϵ
      spl.info[:ϵ_bar], spl.info[:H_bar] = ϵ_bar, H_bar
    elseif m == spl.alg.n_adapt
      dprintln(0, " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption.")
      spl.info[:ϵ] = spl.info[:ϵ_bar]
    end

    push!(spl.info[:accept_his], true)

    vi[spl] = θ
    setlogp!(vi, logp)

    θ_new = realpart(vi[spl])                                         # x_t
    spl.info[:θ_num] += 1
    t = spl.info[:θ_num]                                              # t
    θ_mean_old = copy(spl.info[:θ_mean])                              # x_bar_t-1
    spl.info[:θ_mean] = (t - 1) / t * spl.info[:θ_mean] + θ_new / t   # x_bar_t
    θ_mean_new = spl.info[:θ_mean]                                    # x_bar_t

    if t == 2
      first_two = [θ_mean_old'; θ_new'] # θ_mean_old here only contains the first θ
      spl.info[:θ_vars] = diag(cov(first_two))
    elseif t <= 1000
      # D = length(θ_new)
      D = 2.4^2
      spl.info[:θ_vars] = (t - 1) / t * spl.info[:θ_vars] +
                          (2.4^2 / D) / t * (t * θ_mean_old .* θ_mean_old - (t + 1) * θ_mean_new .* θ_mean_new + θ_new .* θ_new)
    end

    if t > 500
      spl.info[:stds] = sqrt(spl.info[:θ_vars])
      spl.info[:stds] = spl.info[:stds] / min(spl.info[:stds]...)
    end

    dprintln(3, "R -> X...")
    if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

    vi
  end
end

function build_tree(θ::Vector, r::Vector, logu::Float64, v::Int, j::Int, ϵ::Float64, H0::Float64,
                    model::Function, spl::Sampler, vi::VarInfo)
    doc"""
      - θ     : model parameter
      - r     : momentum variable
      - logu  : slice variable (in log scale)
      - v     : direction ∈ {-1, 1}
      - j     : depth of tree
      - ϵ     : leapfrog step size
      - H0    : initial H
    """
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′, τ_valid = leapfrog2(θ, r, 1, v * ϵ, model, vi, spl)
      # Use old H to save computation
      H′ = τ_valid == 0 ? Inf : find_H(r′, model, vi, spl)
      n′ = (logu <= -H′) ? 1 : 0
      s′ = (logu < Δ_max + -H′) ? 1 : 0
      α′ = exp(min(0, -H′ - (-H0)))

      θ′, r′, θ′, r′, θ′, getlogp(vi), n′, s′, α′, 1
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, logp′, n′, s′, α′, n′_α = build_tree(θ, r, logu, v, j - 1, ϵ, H0, model, spl, vi)

      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, logp′′, n′′, s′′, α′′, n′′_α = build_tree(θm, rm, logu, v, j - 1, ϵ, H0, model, spl, vi)
        else
          _, _, θp, rp, θ′′, logp′′, n′′, s′′, α′′, n′′_α = build_tree(θp, rp, logu, v, j - 1, ϵ, H0, model, spl, vi)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
          logp′ = logp′′
        end
        α′ = α′ + α′′
        n′_α = n′_α + n′′_α
        s′ = s′′ * (dot(θp - θm, rm) >= 0 ? 1 : 0) * (dot(θp - θm, rp) >= 0 ? 1 : 0)
        n′ = n′ + n′′
      end

      θm, rm, θp, rp, θ′, logp′, n′, s′, α′, n′_α
    end
  end
