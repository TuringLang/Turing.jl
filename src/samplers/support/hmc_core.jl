function runmodel(model, vi, spl)
  vi.index = 0
  model(vi=vi, sampler=spl) # run model\
end

# Half momentum step
function half_momentum_step(p, ϵ, val∇E)
  dprintln(3, "half_momentum_step...")
  for k in keys(val∇E)
    p[k] -= ϵ * val∇E[k] / 2
  end
  p
end

# Leapfrog step
function leapfrog(values, val∇E, p, ϵ, model, spl)
  dprintln(3, "leapfrog...")

  p = half_momentum_step(p, ϵ, val∇E) # half step for momentum
  for k in keys(val∇E)                # full step for state
    values[k] = Vector{Dual}(values[k] + ϵ * p[k])
  end
  val∇E = gradient(values, model, spl)
  p = half_momentum_step(p, ϵ, val∇E) # half step for momentum

  # Return updated θ and momentum
  values, val∇E, p
end

# Find logjoint
# NOTE: it returns logjoint but not -logjoint
function find_logjoint(model, values, spl)
  values = runmodel(model, values, spl)
  logjoint = values.logjoint        # get logjoint
  values.logjoint = Dual(0)         # reset logjoint
  logjoint
end

# Compute Hamiltonian
function find_H(p, model, values, spl)
  H = 0
  for k in keys(p)
    H += p[k]' * p[k] / 2
  end
  H += realpart(-find_logjoint(model, values, spl))
  H[1]  # Vector{Any, 1} -> Any
end
