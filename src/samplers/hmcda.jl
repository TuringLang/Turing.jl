immutable HMCDA <: InferenceAlgorithm
  n_samples ::  Int       # number of samples
  n_adapt   ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  lambda    ::  Float64   # target leapfrog length
  space     ::  Set       # sampling space, emtpy means all
  group_id  ::  Int
  HMCDA(delta::Float64, lambda::Float64, space...) = HMCDA(1, delta, lambda, space...)
  HMCDA(n_samples::Int, delta::Float64, lambda::Float64) = HMCDA(n_samples, delta, lambda, Set())
  HMCDA(n_samples::Int, delta::Float64, lambda::Float64, space...) = begin
    n_adapt_default = Int(round(n_samples / 5))
    HMCDA(n_samples, n_adapt_default > 1000 ? 1000 : n_adapt_default, delta, lambda, space)
  end
  HMCDA(alg::HMCDA, new_group_id::Int) =
    HMCDA(alg.n_samples, alg.n_adapt, alg.lf_size, alg.lf_num, alg.space, new_group_id)
  HMCDA(n_samples::Int, n_adapt::Int, delta::Float64, lambda::Float64, space...) =
    new(n_samples, n_adapt, delta, lambda, isa(space, Symbol) ? Set([space]) : Set(space), 0)

end

function find_good_eps(model::Function, spl::Sampler{HMCDA}, vi::VarInfo)
  ϵ, p = 1.0, sample_momentum(vi, spl)                # set initial epsilon and momentums
  jointd = exp(-find_H(p, model, vi, spl))  # calculate p(Θ, p) = exp(-H(Θ, p))

  grad = gradient(vi, model, spl)                             # get inital gradient dictionary
  vi_prime, _, p_prime = leapfrog(deepcopy(vi), grad, deepcopy(p), ϵ, model, spl) # make a leapfrog dictionary

  jointd_prime = exp(-find_H(p_prime, model, vi_prime, spl))  # calculate new p(Θ, p)

  # println("[HMCDA] jointd: ", jointd)
  # println("[HMCDA] jointd_prime: ", jointd_prime)

  # Heuristically find optimal ϵ
  a = 2.0 * (jointd_prime / jointd > 0.5 ? 1 : 0) - 1
  while (jointd_prime / jointd)^a > 2.0^(-a)
    # println("[HMCDA] current ϵ:", ϵ)
    ϵ = 2.0^a * ϵ
    vi_prime, _, p_prime = leapfrog(deepcopy(vi), grad, deepcopy(p), ϵ, model, spl)
    jointd_prime = exp(-find_H(p_prime, model, vi_prime, spl))
  end

  ϵ
end

function step(model, spl::Sampler{HMCDA}, vi::VarInfo, is_first::Bool)
  if is_first

    # Run the model for the first time
    dprintln(2, "initialising...")
    vi = runmodel(model, vi, spl)

    vi = link(vi, spl)

    # Heuristically find optimal ϵ
    # println("[HMCDA] finding for ϵ")
    ϵ = find_good_eps(model, spl, vi)
    dprintln(2, "[HMCDA] found initial ϵ:", ϵ)

    vi = invlink(vi, spl)

    spl.info[:ϵ] = ϵ
    spl.info[:μ] = log(10 * ϵ)
    spl.info[:ϵ_bar] = 1.0
    spl.info[:H_bar] = 0.0
    spl.info[:m] = 0

    true, vi
  else
    # Set parameters
    δ = spl.alg.delta
    λ = spl.alg.lambda
    ϵ = spl.info[:ϵ]

    dprintln(2, "current ϵ: $ϵ")
    μ, γ, t_0, κ = spl.info[:μ], 0.05, 10, 0.75
    ϵ_bar, H_bar = spl.info[:ϵ_bar], spl.info[:H_bar]

    dprintln(2, "sampling momentum...")
    p = sample_momentum(vi, spl)

    dprintln(3, "X -> R...")
    vi = link(vi, spl)

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, vi, spl)

    dprintln(3, "first gradient...")
    grad = gradient(vi, model, spl)

    τ = max(1, round(λ / ϵ))
    dprintln(2, "leapfrog for $τ steps")
    for t in 1:τ  # do 'leapfrog' for each var
      vi, grad, p = leapfrog(vi, grad, p, ϵ, model, spl)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, model, vi, spl)

    dprintln(3, "R -> X...")
    vi = invlink(vi, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    realpart!(vi)

    α = min(1, exp(-ΔH))  # MH accept rate

    # Use Dual Averaging to adapt ϵ
    m = spl.info[:m] += 1
    if m <= spl.alg.n_adapt
      H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - α)
      ϵ = exp(μ - sqrt(m) / γ * H_bar)
      ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
      spl.info[:ϵ] = ϵ
      spl.info[:ϵ_bar], spl.info[:H_bar] = ϵ_bar, H_bar
    else
      spl.info[:ϵ] = spl.info[:ϵ_bar]
    end

    dprintln(2, "decide wether to accept...")
    if rand() < α      # accepted
      true, vi
    else                                # rejected
      false, vi
    end
  end
end
