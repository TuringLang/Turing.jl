global Δ_max = 1000

doc"""
Calculate dot(θp - θm, r)
"""
function direction(θm, θp, r, model, spl)
  s = 0
  for k in keys(r)
    s += dot(θp[k] - θm[k], r[k])
  end
  s
end

setchunksize(chun_size::Int) = global CHUNKSIZE = chunk_size

function runmodel(model, _vi, spl, default_logp=0.0)
  dprintln(4, "run model...")
  vi = deepcopy(_vi)
  vi.logp = default_logp
  vi.index = 0
  model(vi=vi, sampler=spl) # run model\
end

function sample_momentum(vi::VarInfo, spl)
  dprintln(2, "sampling momentum...")
  kys = keys(vi)
  if ~isempty(spl.alg.space)
    filter!(k -> getsym(vi, k) in spl.alg.space, kys)
  end
  Dict(uid(k) => randn(length(vi[k])) for k in kys)
end

# Half momentum step
function half_momentum_step!(p, ϵ, grad)
  dprintln(3, "half_momentum_step...")
  for k in keys(grad)
    p[k] -= ϵ * grad[k] / 2
  end
end

# Leapfrog step
function leapfrog(_vi, _p, τ, ϵ, model, spl)

  reject = false
  vi = deepcopy(_vi)
  p = deepcopy(_p)

  dprintln(3, "first gradient...")
  grad = gradient(vi, model, spl)
  # Verify gradients; reject if gradients is NaN or Inf.
  verifygrad(grad) || (reject = true)

  dprintln(2, "leapfrog stepping...")
  for t in 1:τ  # do 'leapfrog' for each var
    half_momentum_step!(p, ϵ, grad) # half step for momentum

    duplicate!(vi)

    for k in keys(grad)                # full step for state
      range = getrange(vi, k)
      vi[range] = vi[range] + ϵ * p[k]
    end

    grad = gradient(vi, model, spl)

    # Verify gradients; reject if gradients is NaN or Inf.
    verifygrad(grad) || (reject = true; break)

    half_momentum_step!(p, ϵ, grad) # half step for momentum

    if realpart(vi.logp) == -Inf
      break
    elseif isnan(realpart(vi.logp)) || realpart(vi.logp) == Inf
      dwarn(0, "Numerical error: vi.lojoint = $(vi.logp)")
      reject = true
      break
    end
  end

  keeplast!(vi)

  # Return updated θ and momentum
  vi, p, reject
end

# Compute Hamiltonian
function find_H(p, model, _vi, spl)
  vi = deepcopy(_vi)
  H = 0
  for k in keys(p)
    H += dot(p[k], p[k]) / 2
  end
  vi = runmodel(model, vi, spl)
  H += realpart(-vi.logp)
  H = H[1]  # Vector{Any, 1} -> Any
  if isnan(H) || isinf(H); H = Inf else H end
end

function find_good_eps{T}(model::Function, spl::Sampler{T}, vi::VarInfo)
  ϵ, p = 1.0, sample_momentum(vi, spl)    # set initial epsilon and momentums
  log_p_r_Θ = -find_H(p, model, vi, spl)  # calculate p(Θ, r) = exp(-H(Θ, r))

  # Make a leapfrog step until accept
  vi_prime, p_prime, reject = leapfrog(vi, p, 1, ϵ, model, spl)
  while reject
    ϵ *= 0.5
    vi_prime, p_prime, reject = leapfrog(vi, p, 1, ϵ, model, spl)
  end
  ϵ_bar = ϵ
  log_p_r_Θ′ = -find_H(p_prime, model, vi_prime, spl)   # calculate new p(Θ, p)

  # Heuristically find optimal ϵ
  a = 2.0 * (log_p_r_Θ′ - log_p_r_Θ > log(0.5) ? 1 : 0) - 1
  while (exp(log_p_r_Θ′ - log_p_r_Θ))^a > 2.0^(-a)
    ϵ = 2.0^a * ϵ
    vi_prime, p_prime = leapfrog(vi, p, 1, ϵ, model, spl)
    log_p_r_Θ′ = -find_H(p_prime, model, vi_prime, spl)
  end

  println("[$T] found initial ϵ: ", ϵ)
  ϵ_bar, ϵ
end
