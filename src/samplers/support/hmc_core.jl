setchunksize(chun_size::Int) = global CHUNKSIZE = chunk_size

function runmodel(model, vi, spl)
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

  dprintln(2, "leapfrog stepping...")
  for t in 1:τ  # do 'leapfrog' for each var
    if haskey(spl.info, :m) && spl.info[:m] > spl.alg.n_adapt
      dprintln(2, "[Turing]: p = $p")
      dprintln(2, "[Turing]: vi = $vi")
    end

    for k in keys(grad)
      if any(isnan(grad[k])) || any(isinf(grad[k]))
        warn("[Turing]: grad = $(grad)")
        reject = true
        break
      end
    end
    p = half_momentum_step(p, ϵ, grad) # half step for momentum
    for k in keys(grad)                # full step for state
      # NOTE: Vector{Dual} is necessary magic conversion
      vi[k] = Vector{Dual}(vi[k] + ϵ * p[k])
    end
    grad = gradient(vi, model, spl)
    for k in keys(grad)
      if any(isnan(grad[k])) || any(isinf(grad[k]))
        warn("[Turing]: grad = $(grad)")
        reject = true
        break
      end
    end
    p = half_momentum_step(p, ϵ, grad) # half step for momentum
    if realpart(vi.logjoint) == -Inf
      break
    elseif isnan(realpart(vi.logjoint)) || realpart(vi.logjoint) == Inf
      warn("[Turing]: Numerical error: vi.lojoint = $(vi.logjoint)")
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
  vi = runmodel(model, vi, spl)
  logjoint = vi.logjoint        # get logjoint
  vi.logjoint = Dual(0)         # reset logjoint
  logjoint
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
