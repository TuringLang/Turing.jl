"""
    NUTS(n_iters::Int, n_adapts::Int, delta::Float64)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.6j_max)
```

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `n_adapts::Int` : The number of samples to use with adapatation.
- `delta::Float64` : Target acceptance rate.

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

sample(gdemo([1.j_max, 2]), NUTS(1000, 200, 0.6j_max))
```
"""
mutable struct NUTS{AD, space} <: AdaptiveHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    n_adapts  ::  Int       # number of samples with adaption for epsilon
    delta     ::  Float64   # target accept rate
    gid       ::  Int       # group ID
end
function NUTS{AD}(n_adapts::Int, delta::Float64, space...) where AD
    NUTS{AD, space}(1, n_adapts, delta, 0)
end
function NUTS{AD}(n_iters::Int, n_adapts::Int, delta::Float64, space...) where AD
    NUTS{AD, space}(n_iters, n_adapts, delta, 0)
end
function NUTS{AD}(n_iters::Int, delta::Float64) where AD
    n_adapts_default = Int(round(n_iters / 2))
    NUTS{AD, ()}(n_iters, n_adapts_default > 1000 ? 1000 : n_adapts_default, delta, 0)
end
function NUTS{AD1}(alg::NUTS{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    NUTS{AD1, space}(alg.n_iters, alg.n_adapts, alg.delta, new_gid)
end
function NUTS{AD, space}(alg::NUTS, new_gid::Int) where {AD, space}
    NUTS{AD, space}(alg.n_iters, alg.n_adapts, alg.delta, new_gid)
end
