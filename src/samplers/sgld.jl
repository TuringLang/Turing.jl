doc"""
    SGLD(n_iters::Int, step_size::Float64)

 Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Example:

```julia
@model example begin
  ...
end

sample(example, SGLD(1000, 0.5))
```
"""
immutable SGLD <: Hamiltonian
  n_iters   ::  Int       # number of samples
  step_size ::  Float64   # constant scale factor of learning rate
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int
  function SGLD(step_size::Float64, space...)
    SGLD(1, step_size, space..., 0)
  end
  function SGLD(n_iters, step_size)
    new(n_iters, step_size, Set(), 0)
  end
  function SGLD(n_iters, step_size, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_iters, step_size, space, 0)
  end
  SGLD(alg::SGLD, new_gid::Int) = new(alg.n_iters, alg.step_size, alg.space, new_gid)
end

function step(model, spl::Sampler{SGLD}, vi::VarInfo, is_first::Bool)
  if is_first
    if ~haskey(spl.info, :wum)
      if spl.alg.gid != 0 link!(vi, spl) end    # X -> R

      wum = WarmUpManager(1, 1, Dict())
      wum[:ϵ] = [spl.alg.step_size]
      wum[:stds] = ones(length(vi[spl]))
      spl.info[:wum] = wum

      oldθ = realpart(vi[spl])
      vi[spl] = oldθ

      # Initialize iteration counter
      spl.info[:t] = 0

      if spl.alg.gid != 0 invlink!(vi, spl) end # R -> X
    end

    push!(spl.info[:accept_his], true)

    vi
  else
    # Update iteration counter
    t = deepcopy(spl.info[:t]) + 1
    spl.info[:t] = deepcopy(t)

    dprintln(2, "compute current step size...")
    γ = .35
    ϵ_t = spl.alg.step_size / t^γ # NOTE: Choose γ=.55 in paper
    push!(spl.info[:wum][:ϵ], ϵ_t)

    dprintln(3, "X-> R...")
    if spl.alg.gid != 0
      link!(vi, spl)
      runmodel(model, vi, spl)
    end

    dprintln(2, "recording old variables...")
    grad = gradient(vi, model, spl)
    old_θ = realpart(vi[spl])
    θ = deepcopy(old_θ)

    dprintln(2, "update latent variables...")
    v = zeros(Float64, size(old_θ))
    for k in 1:size(old_θ, 1)
      noise = rand(MvNormal(zeros(length(old_θ[k,:])), sqrt(ϵ_t)*ones(length(old_θ[k,:]))))
      θ[k,:] = old_θ[k,:] - 0.5 * ϵ_t * grad[k,:] + noise
    end


    dprintln(2, "always accept...")
    push!(spl.info[:accept_his], true)
    vi[spl] = θ

    dprintln(3, "R -> X...")
    if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

    vi
  end
end
