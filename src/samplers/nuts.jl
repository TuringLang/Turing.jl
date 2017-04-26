immutable NUTS <: InferenceAlgorithm
  n_samples ::  Int       # number of samples
  n_adapt   ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  space     ::  Set       # sampling space, emtpy means all
  group_id  ::  Int

  NUTS(n_adapt::Int, delta::Float64, space...) = new(1, isa(space, Symbol) ? Set([space]) : Set(space), delta, Set(), 0)
  NUTS(n_samples::Int, n_adapt::Int, delta::Float64, space...) = new(n_samples, n_adapt, delta, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  NUTS(n_samples::Int, delta::Float64) = begin
    n_adapt_default = Int(round(n_samples / 5))
    new(n_samples, n_adapt_default > 1000 ? 1000 : n_adapt_default, delta, Set(), 0)
  end
  NUTS(alg::NUTS, new_group_id::Int) = new(alg.n_samples, alg.n_adapt, alg.delta, alg.space, new_group_id)
end

function step(model, spl::Sampler{NUTS}, vi::VarInfo, is_first::Bool)
  if is_first
    vi_0 = deepcopy(vi)

    vi = link(vi, spl)

    # Heuristically find optimal ϵ
    # println("[HMCDA] finding for ϵ")
    ϵ_bar, ϵ = find_good_eps(model, spl, vi)

    vi = invlink(vi, spl)

    spl.info[:ϵ] = ϵ
    spl.info[:μ] = log(10 * ϵ)
    # spl.info[:ϵ_bar] = 1.0
    spl.info[:ϵ_bar] = ϵ_bar  # NOTE: is this correct?
    spl.info[:H_bar] = 0.0
    spl.info[:m] = 0

    true, vi_0
  else
    # Set parameters
    δ = spl.alg.delta
    ϵ = spl.info[:ϵ]

    dprintln(2, "current ϵ: $ϵ")
    μ, γ, t_0, κ = spl.info[:μ], 0.05, 10, 0.75
    ϵ_bar, H_bar = spl.info[:ϵ_bar], spl.info[:H_bar]

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(3, "X -> R...")
    vi = link(vi, spl)

    dprintln(3, "sample slice variable u")
    u = rand() * exp(-find_H(p, model, vi, spl))

    θm, θp, rm, rp, j, vi_new, n, s = deepcopy(vi), deepcopy(vi), deepcopy(p), deepcopy(p), 0, deepcopy(vi), 1, 1
    local α, n_α
    while s == 1
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, θ′, n′, s′, α, n_α = build_tree(θm, rm, u, v_j, j, ϵ, vi, p, model, spl)
      else
        _, _, θp, rp, θ′, n′, s′, α, n_α = build_tree(θp, rp, u, v_j, j, ϵ, vi, p, model, spl)
      end
      if s′ == 1
        if rand() < min(1, n′ / n)
          vi_new = θ′
        end
      end
      n = n + n′
      s = s′ & (direction(θm, θp, rm, model, spl) >= 0) & (direction(θm, θp, rp, model, spl) >= 0)
      j = j + 1
    end

    dprintln(3, "R -> X...")
    vi = invlink(vi, spl)

    cleandual!(vi)

    # Use Dual Averaging to adapt ϵ
    m = spl.info[:m] += 1
    if m <= spl.alg.n_adapt
      H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - α / n_α)
      ϵ = exp(μ - sqrt(m) / γ * H_bar)
      ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
      spl.info[:ϵ] = ϵ
      spl.info[:ϵ_bar], spl.info[:H_bar] = ϵ_bar, H_bar
    else
      spl.info[:ϵ] = spl.info[:ϵ_bar]
    end

    true, vi_new
  end
end

function build_tree(θ, r, u, v, j, ϵ, θ0, r0, model, spl)
    doc"""
      - θ   : model parameter
      - r   : momentum variable
      - u   : slice variable
      - v   : direction ∈ {-1, 1}
      - j   : depth of tree
      - ϵ   : leapfrog step size
      - θ0  : initial model parameter
      - r0  : initial mometum variable
    """
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, 1, v * ϵ, model, spl)
      n′ = u <= exp(-find_H(r′, model, θ′, spl))
      s′ = u < exp(Δ_max - find_H(r′, model, θ′, spl))
      return θ′, r′, θ′, r′, θ′, n′, s′, min(1, exp(-find_H(r′, model, θ′, spl) - (-find_H(r0, model, θ0, spl)))), 1
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′, α′, n′_α = build_tree(θ, r, u, v, j - 1, ϵ, θ0, r0, model, spl)
      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′, α′′, n′′_α = build_tree(θm, rm, u, v, j - 1, ϵ, θ0, r0, model, spl)
        else
          _, _, θp, rp, θ′′, n′′, s′′, α′′, n′′_α = build_tree(θp, rp, u, v, j - 1, ϵ, θ0, r0, model, spl)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        α′ = α′ + α′′
        n′_α = n′_α + n′′_α
        s′ = s′′ & (direction(θm, θp, rm, model, spl) >= 0) & (direction(θm, θp, rp, model, spl) >= 0)
        n′ = n′ + n′′
      end
      return θm, rm, θp, rp, θ′, n′, s′, α′, n′_α
    end
  end
