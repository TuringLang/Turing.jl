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
mutable struct NUTS{AD, T} <: AdaptiveHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    n_adapts  ::  Int       # number of samples with adaption for epsilon
    delta     ::  Float64   # target accept rate
    space     ::  Set{T}    # sampling space, emtpy means all
end
NUTS(args...; kwargs...) = NUTS{ADBackend()}(args...; kwargs...)
function NUTS{AD}(n_adapts::Int, delta::Float64, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    NUTS{AD, eltype(_space)}(1, n_adapts, delta, _space)
end
function NUTS{AD}(n_iters::Int, n_adapts::Int, delta::Float64, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    NUTS{AD, eltype(_space)}(n_iters, n_adapts, delta, _space)
end
function NUTS{AD}(n_iters::Int, delta::Float64) where AD
    n_adapts_default = Int(round(n_iters / 2))
    NUTS{AD, Any}(n_iters, n_adapts_default > 1000 ? 1000 : n_adapts_default, delta, Set())
end

function hmc_step(θ, lj, logπ, _∂logπ∂θ, ϵ, alg::NUTS, metric)
    h = AdvancedHMC.Hamiltonian(metric, logπ, x->_∂logπ∂θ(x)[2])
    max_depth = 5
    Δ_max = 1000.0
    prop = AdvancedHMC.NUTS(AdvancedHMC.Leapfrog(ϵ), max_depth, Δ_max)

    r = AdvancedHMC.rand_momentum(h)
    θ_new, _, α, _ = AdvancedHMC.transition(prop, h, Vector{Float64}(θ), r)

    lj_new = logπ(θ_new)
    is_accept = true
    return θ_new, lj_new, is_accept, α
end
