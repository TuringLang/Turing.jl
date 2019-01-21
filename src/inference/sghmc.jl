"""
    SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)

Stochastic Gradient Hamiltonian Monte Carlo sampler.

Usage:

```julia
SGHMC(1000, 0.01, 0.1)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
- `learning_rate::Float64` : The learning rate.
- `momentum_decay::Float64` : Momentum decay variable.

Example:

```julia
@model example begin
  ...
end

sample(example, SGHMC(1000, 0.01, 0.1))
```
"""
mutable struct SGHMC{AD, T} <: StaticHamiltonian{AD}
    n_iters::Int       # number of samples
    learning_rate::Float64   # learning rate
    momentum_decay::Float64   # momentum decay
    space::Set{T}    # sampling space, emtpy means all
    gid::Int
end
SGHMC(args...) = SGHMC{ADBackend()}(args...)
function SGHMC{AD}(learning_rate::Float64, momentum_decay::Float64, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SGHMC{AD, eltype(_space)}(1, learning_rate, momentum_decay, _space, 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay) where AD
    return SGHMC{AD, Any}(n_iters, learning_rate, momentum_decay, Set(), 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SGHMC{AD, eltype(_space)}(n_iters, learning_rate, momentum_decay, _space, 0)
end
function SGHMC{AD1}(alg::SGHMC{AD2, T}, new_gid::Int) where {AD1, AD2, T}
    return SGHMC{AD1, T}(alg.n_iters, alg.learning_rate, alg.momentum_decay, alg.space, new_gid)
end
function SGHMC{AD, T}(alg::SGHMC, new_gid::Int) where {AD, T}
    return SGHMC{AD, T}(alg.n_iters, alg.learning_rate, alg.momentum_decay, alg.space, new_gid)
end

function step(model, spl::Sampler{<:SGHMC}, vi::VarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)

    # Initialize velocity
    v = zeros(Float64, size(vi[spl]))
    spl.info[:v] = v

    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGHMC}, vi::VarInfo, is_first::Val{false})
    # Set parameters
    η, α = spl.alg.learning_rate, spl.alg.momentum_decay

    @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    @debug "recording old variables..."
    θ, v = vi[spl], spl.info[:v]
    _, grad = gradient(θ, vi, model, spl)
    verifygrad(grad)

    # Implements the update equations from (15) of Chen et al. (2014).
    @debug "update latent variables and velocity..."
    θ .+= v
    v .= (1 - α) .* v .- η .* grad .+ rand.(Normal.(zeros(length(θ)), sqrt(2 * η * α)))

    @debug "saving new latent variables..."
    vi[spl] = θ

    @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    @debug "always accept..."
    return vi, true
end
