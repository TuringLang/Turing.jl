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

function runmodel(model, _vi, spl)
  vi = deepcopy(_vi)
  vi.index = 0
  model(vi=vi, sampler=spl) # run model\
end

function sample_momentum(vi::VarInfo, spl)
  dprintln(2, "sampling momentum...")
  p = Dict(uid(k) => randn(length(vi[k])) for k in keys(vi))
  if ~isempty(spl.alg.space)
    p = filter((k, p) -> getsym(vi, k) in spl.alg.space, p)
  end
  p
end

# Half momentum step
function half_momentum_step(_p, ϵ, val∇E)
  p = deepcopy(_p)
  dprintln(3, "half_momentum_step...")
  for k in keys(val∇E)
    p[k] -= ϵ * val∇E[k] / 2
  end
  p
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
    p = half_momentum_step(p, ϵ, grad) # half step for momentum
    for k in keys(grad)                # full step for state
      val_vec = vi[k]
      for i = 1:length(val_vec)
        val_vec[i] = val_vec[i] + ϵ * p[k][i]
      end
    end
    grad = gradient(vi, model, spl)

    # Verify gradients; reject if gradients is NaN or Inf.
    verifygrad(grad) || (reject = true; break)

    p = half_momentum_step(p, ϵ, grad) # half step for momentum
    if realpart(vi.logjoint) == -Inf
      break
    elseif isnan(realpart(vi.logjoint)) || realpart(vi.logjoint) == Inf
      dwarn(0, "Numerical error: vi.lojoint = $(vi.logjoint)")
      reject = true
      break
    end
  end

  # Return updated θ and momentum
  vi, p, reject
end

# Find logjoint
# NOTE: it returns logjoint but not -logjoint
function find_logjoint(model, _vi, spl)
  vi = deepcopy(_vi)
  vi.logjoint = 0
  vi = runmodel(model, vi, spl)
  vi.logjoint        # get logjoint
end

# Compute Hamiltonian
function find_H(p, model, vi, spl)
  H = 0
  for k in keys(p)
    H += p[k]' * p[k] / 2
  end
  H += realpart(-find_logjoint(model, vi, spl))
  H = H[1]  # Vector{Any, 1} -> Any
  if isnan(H) || isinf(H); H = Inf else H end
end

function find_good_eps{T}(model::Function, spl::Sampler{T}, vi::VarInfo)
  ϵ, p = 1.0, sample_momentum(vi, spl)                # set initial epsilon and momentums
  jointd = exp(-find_H(p, model, vi, spl))  # calculate p(Θ, p) = exp(-H(Θ, p))

  # println("[HMCDA] grad: ", grad)
  # println("[HMCDA] p: ", p)
  # println("[HMCDA] vi: ", vi)
  vi_prime, p_prime = leapfrog(vi, p, 1, ϵ, model, spl) # make a leapfrog dictionary

  jointd_prime = exp(-find_H(p_prime, model, vi_prime, spl))  # calculate new p(Θ, p)

  # println("[HMCDA] jointd: ", jointd)
  # println("[HMCDA] jointd_prime: ", jointd_prime)

  # This trick prevents the log-joint or its graident from being infinte
  # Ref: https://github.com/mfouesneau/NUTS/blob/master/nuts.py#L111
  # QUES: will this lead to some bias of the sampler?
  while isnan(jointd_prime)
    ϵ *= 0.5
    # println("[HMCDA] current ϵ: ", ϵ)
    # println("[HMCDA] jointd_prime: ", jointd_prime)
    # println("[HMCDA] vi_prime: ", vi_prime)
    vi_prime, p_prime = leapfrog(vi, p, 1, ϵ, model, spl)
    jointd_prime = exp(-find_H(p_prime, model, vi_prime, spl))
  end
  ϵ_bar = ϵ

  # Heuristically find optimal ϵ
  a = 2.0 * (jointd_prime / jointd > 0.5 ? 1 : 0) - 1
  while (jointd_prime / jointd)^a > 2.0^(-a)
    # println("[HMCDA] current ϵ: ", ϵ)
    # println("[HMCDA] jointd_prime: ", jointd_prime)
    # println("[HMCDA] vi_prime: ", vi_prime)
    ϵ = 2.0^a * ϵ
    vi_prime, p_prime = leapfrog(vi, p, 1, ϵ, model, spl)
    jointd_prime = exp(-find_H(p_prime, model, vi_prime, spl))
  end

  println("[$T] found initial ϵ: ", ϵ)
  ϵ_bar, ϵ
end
