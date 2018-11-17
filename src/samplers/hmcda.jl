"""
    HMCDA(n_iters::Int, n_adapts::Int, delta::Float64, lambda::Float64)

Hamiltonian Monte Carlo sampler wiht Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))
```
"""
mutable struct HMCDA{T} <: AdaptiveHamiltonian
  n_iters   ::  Int       # number of samples
  n_adapts  ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  lambda    ::  Float64   # target leapfrog length
  space     ::  Set{T}    # sampling space, emtpy means all
  gid       ::  Int       # group ID
end
function HMCDA(n_adapts::Int, delta::Float64, lambda::Float64, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMCDA(1, n_adapts, delta, lambda, _space, 0)
end
function HMCDA(n_iters::Int, delta::Float64, lambda::Float64)
    n_adapts_default = Int(round(n_iters / 2))
    n_adapts = n_adapts_default > 1000 ? 1000 : n_adapts_default
    return HMCDA(n_iters, n_adapts, delta, lambda, Set(), 0)
end
function HMCDA(alg::HMCDA, new_gid::Int)
    return HMCDA(alg.n_iters, alg.n_adapts, alg.delta, alg.lambda, alg.space, new_gid)
end
HMCDA{T}(alg::HMCDA, new_gid::Int) where {T} = HMCDA(alg, new_gid)
function HMCDA(n_iters::Int, n_adapts::Int, delta::Float64, lambda::Float64)
    return HMCDA(n_iters, n_adapts, delta, lambda, Set(), 0)
end
function HMCDA(n_iters::Int, n_adapts::Int, delta::Float64, lambda::Float64, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMCDA(n_iters, n_adapts, delta, lambda, _space, 0)
end

function hmc_step(θ, lj, lj_func, grad_func, ϵ, std, alg::HMCDA; rev_func=nothing, log_func=nothing)
  θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(
            θ, lj, lj_func, grad_func, ϵ, alg.lambda, std; rev_func=rev_func, log_func=log_func)
  return θ_new, lj_new, is_accept, α
end
