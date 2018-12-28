
"""
    SGLD(n_iters::Int, epsilon::Float64)

 Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
- `epsilon::Float64` : The scaling factor for the learing rate.

Example:

```julia
@model example begin
  ...
end

sample(example, SGLD(1000, 0.5))
```
"""
mutable struct SGLD{AD, space} <: StaticHamiltonian{AD}
    n_iters :: Int       # number of samples
    epsilon :: Float64   # constant scale factor of learning rate
    gid     :: Int
end
function SGLD{AD}(epsilon::Float64, space...) where AD 
    SGLD{AD, space}(1, epsilon, 0)
end
function SGLD{AD}(n_iters, epsilon) where AD
    SGLD{AD, ()}(n_iters, epsilon, 0)
end
function SGLD{AD}(n_iters, epsilon, space...) where AD
    return SGLD{AD, space}(n_iters, epsilon, 0)
end
function SGLD{AD1}(alg::SGLD{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    SGLD{AD1, space}(alg.n_iters, alg.epsilon, new_gid)
end
function SGLD{AD, space}(alg::SGLD, new_gid::Int) where {AD, space}
    SGLD{AD, space}(alg.n_iters, alg.epsilon, new_gid)
end
