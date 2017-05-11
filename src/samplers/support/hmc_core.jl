global Δ_max = 1000

setchunksize(chun_size::Int) = global CHUNKSIZE = chunk_size

runmodel(model::Function, vi::VarInfo, spl::Union{Void,Sampler}) = begin
  dprintln(4, "run model...")
  vi.index = 0
  model(vi=vi, sampler=spl) # run model
end

sample_momentum(vi::VarInfo, spl::Sampler) = begin
  dprintln(2, "sampling momentum...")
  randn(length(getranges(vi, spl)))
end

# Leapfrog step
leapfrog(_vi::VarInfo, _p::Vector, τ::Int, ϵ::Float64, model::Function, spl::Sampler) = begin

  vi = deepcopy(_vi)
  p = deepcopy(_p)

  dprintln(3, "first gradient...")
  grad = gradient2(vi, model, spl)
  # Verify gradients; reject if gradients is NaN or Inf.
  verifygrad(grad) || (return vi, p, 0)

  dprintln(2, "leapfrog stepping...")
  τ_valid = 0; p_old = deepcopy(p)
  for t in 1:τ        # do 'leapfrog' for each var
    p_old[1:end] = p[1:end]

    p -= ϵ * grad / 2

    expand!(vi)

    vi[spl] += ϵ * p  # full step for state

    grad = gradient2(vi, model, spl)

    # Verify gradients; reject if gradients is NaN or Inf
    verifygrad(grad) || (shrink!(vi); p = p_old; break)

    p -= ϵ * grad / 2

    τ_valid += 1
  end

  last!(vi)

  # Return updated θ and momentum
  vi, p, τ_valid
end

# Compute Hamiltonian
find_H(p::Vector, model::Function, vi::VarInfo, spl::Sampler) = begin
  # NOTE: getlogp(vi) = 0 means the current vals[end] hasn't been used at all.
  #       This can be a result of link/invlink (where expand! is used)
  if getlogp(vi) == 0 vi = runmodel(model, vi, spl) end

  H = dot(p, p) / 2 + realpart(-getlogp(vi))
  if isnan(H) H = Inf else H end
end

find_good_eps{T}(model::Function, spl::Sampler{T}, vi::VarInfo) = begin
  ϵ, p = 1.0, sample_momentum(vi, spl)    # set initial epsilon and momentums
  log_p_r_Θ = -find_H(p, model, vi, spl)  # calculate p(Θ, r) = exp(-H(Θ, r))

  # # Make a leapfrog step until accept
  # vi_prime, p_prime, τ_valid = leapfrog(vi, p, 1, ϵ, model, spl)
  # while τ_valid == 0
  #   ϵ *= 0.5
  #   vi_prime, p_prime, τ_valid = leapfrog(vi, p, 1, ϵ, model, spl)
  # end
  # ϵ_bar = ϵ
  vi_prime, p_prime, τ = leapfrog(vi, p, 1, ϵ, model, spl)
  log_p_r_Θ′ = τ == 0 ? -Inf : -find_H(p_prime, model, vi_prime, spl)   # calculate new p(Θ, p)

  # Heuristically find optimal ϵ
  a = 2.0 * (log_p_r_Θ′ - log_p_r_Θ > log(0.5) ? 1 : 0) - 1
  while (exp(log_p_r_Θ′ - log_p_r_Θ))^a > 2.0^(-a)
    ϵ = 2.0^a * ϵ
    vi_prime, p_prime, τ = leapfrog(vi, p, 1, ϵ, model, spl)
    log_p_r_Θ′ = τ == 0 ? -Inf : -find_H(p_prime, model, vi_prime, spl)
    dprintln(1, "a = $a, log_p_r_Θ′ = $log_p_r_Θ′")
  end

  println("\r[$T] found initial ϵ: ", ϵ)
  ϵ
end
