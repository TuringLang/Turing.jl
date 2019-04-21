"""
    HMCDA(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.0)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
- `n_adapts::Int` : Numbers of samples to use for adaptation.
- `δ::Float64` : Target acceptance rate. 65% is often recommended.
- `λ::Float64` : Target leapfrop length.
- `init_ϵ::Float64=0.0` : Inital step size; 0 means automatically search by Turing.

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

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research 15, no. 1 (2014): 1593-1623.
"""
mutable struct HMCDA{AD, T} <: AdaptiveHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    n_adapts  ::  Int       # number of samples with adaption for epsilon
    δ         ::  Float64   # target accept rate
    λ         ::  Float64   # target leapfrog length
    space     ::  Set{T}    # sampling space, emtpy means all
    init_ϵ    ::  Float64
    metricT
end
HMCDA(args...; kwargs...) = HMCDA{ADBackend()}(args...; kwargs...)
function HMCDA{AD}(n_iters::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.0, metricT=AHMC.UnitEuclideanMetric) where AD
    n_adapts_default = Int(round(n_iters / 2))
    n_adapts = n_adapts_default > 1000 ? 1000 : n_adapts_default
    return HMCDA{AD, Any}(n_iters, n_adapts, δ, λ, Set(), init_ϵ, metricT)
end
function HMCDA{AD}(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.0, metricT=AHMC.UnitEuclideanMetric) where AD
    return HMCDA{AD, Any}(n_iters, n_adapts, δ, λ, Set(), init_ϵ, metricT)
end
function HMCDA{AD}(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64, space...; init_ϵ::Float64=0.0, metricT=AHMC.UnitEuclideanMetric) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMCDA{AD, eltype(_space)}(n_iters, n_adapts, δ, λ, _space, init_ϵ, metricT)
end

function hmc_step(θ, lj, lj_func, grad_func, ϵ, alg::HMCDA, metric)
    θ_new, lj_new, is_accept, α = _hmc_step(θ, lj, lj_func, grad_func, ϵ, alg.λ, metric)
    return θ_new, lj_new, is_accept, α
end
