"""
    SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)

Stochastic Gradient Hamiltonian Monte Carlo sampler.

Usage:

```julia
SGHMC(1000, 0.01, 0.1)
```

Example:

```julia
@model example begin
  ...
end

sample(example, SGHMC(1000, 0.01, 0.1))
```
"""
mutable struct SGHMC <: Hamiltonian
  n_iters        ::  Int       # number of samples
  learning_rate  ::  Float64   # learning rate
  momentum_decay ::  Float64   # momentum decay
  space          ::  Set       # sampling space, emtpy means all
  gid            ::  Int
  function SGHMC(learning_rate::Float64, momentum_decay::Float64, space...)
    SGHMC(1, learning_rate, momentum_decay, space..., 0)
  end
  function SGHMC(n_iters, learning_rate, momentum_decay)
    new(n_iters, learning_rate, momentum_decay, Set(), 0)
  end
  function SGHMC(n_iters, learning_rate, momentum_decay, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_iters, learning_rate, momentum_decay, space, 0)
  end
  SGHMC(alg::SGHMC, new_gid::Int) = new(alg.n_iters, alg.learning_rate, alg.momentum_decay, alg.space, new_gid)
end

function step(model, spl::Sampler{SGHMC}, vi::VarInfo, is_first::Bool)
  if is_first
    if ~haskey(spl.info, :wum)
      if spl.alg.gid != 0 link!(vi, spl) end    # X -> R

      #wum = WarmUpManager(1, 1, Dict())
      
      D = length(vi[spl])
      ve = VarEstimator{Float64}(0, zeros(D), zeros(D))
      wum = WarmUpManager(1, Dict(), ve)
      wum[:ϵ] = [spl.alg.learning_rate]
      wum[:stds] = ones(D)
      spl.info[:wum] = wum

      oldθ = realpart(vi[spl])
      vi[spl] = oldθ

      # Initialize velocity
      v = zeros(Float64, size(oldθ))
      spl.info[:v] = v

      if spl.alg.gid != 0 invlink!(vi, spl) end # R -> X
    end

    push!(spl.info[:accept_his], true)

    vi
  else
    # Set parameters
    η, α = spl.alg.learning_rate, spl.alg.momentum_decay

    @debug "X-> R..."
    if spl.alg.gid != 0
      link!(vi, spl)
      runmodel(model, vi, spl)
    end

    @debug "recording old variables..."
    old_θ = realpart(vi[spl]);
    θ = deepcopy(old_θ)
    _, grad = gradient(old_θ, vi, model, spl)
    old_v = deepcopy(spl.info[:v])
    v = deepcopy(old_v)

    if verifygrad(grad)
      @debug "update latent variables and velocity..."
      # Implements the update equations from (15) of Chen et al. (2014).
      for k in 1:size(old_θ, 1)
        θ[k,:] = old_θ[k,:] + old_v[k,:]
        noise = rand(MvNormal(zeros(length(old_θ[k,:])), sqrt.(2 * η * α)*ones(length(old_θ[k,:]))))
        v[k,:] = (1. - α) * old_v[k,:] - η * grad[k,:] + noise # NOTE: divide η by batch size
      end
    end

    @debug "saving new latent variables and velocity..."
    spl.info[:v] = v
    vi[spl] = θ

    @debug "always accept..."
    push!(spl.info[:accept_his], true)

    @debug "R -> X..."
    if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

    vi
  end
end
