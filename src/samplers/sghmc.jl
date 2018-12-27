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
mutable struct SGHMC{AD, space} <: StaticHamiltonian{AD}
    n_iters::Int       # number of samples
    learning_rate::Float64   # learning rate
    momentum_decay::Float64   # momentum decay
    gid::Int
end
function SGHMC{AD}(learning_rate::Float64, momentum_decay::Float64, space...) where AD
    return SGHMC{AD, space}(1, learning_rate, momentum_decay, 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay) where AD
    return SGHMC{AD, ()}(n_iters, learning_rate, momentum_decay, 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay, space...) where AD
    return SGHMC{AD, space}(n_iters, learning_rate, momentum_decay, 0)
end
function SGHMC{AD1}(alg::SGHMC{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    return SGHMC{AD1, space}(alg.n_iters, alg.learning_rate, alg.momentum_decay, new_gid)
end
function SGHMC{AD, space}(alg::SGHMC, new_gid::Int) where {AD, space}
    return SGHMC{AD, space}(alg.n_iters, alg.learning_rate, alg.momentum_decay, new_gid)
end

getspace(::SGHMC{<:Any, space}) where space = space
