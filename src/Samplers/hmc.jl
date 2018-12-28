"""
    HMC(n_iters::Int, epsilon::Float64, tau::Int)

Hamiltonian Monte Carlo sampler.

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `epsilon::Float64` : The leapfrog step size to use.
- `tau::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
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

sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))
```

Tips:

- If you are receiving gradient errors when using `HMC`, try reducing the
`step_size` parameter.

```julia
# Original step_size
sample(gdemo([1.5, 2]), HMC(1000, 0.1, 10))

# Reduced step_size.
sample(gdemo([1.5, 2]), HMC(1000, 0.01, 10))
```
"""
mutable struct HMC{AD, space} <: StaticHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    epsilon   ::  Float64   # leapfrog step size
    tau       ::  Int       # leapfrog step number
    gid       ::  Int       # group ID
end
function HMC{AD}(epsilon::Float64, tau::Int, space...) where AD
    return HMC{AD, space}(1, epsilon, tau, 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int) where AD
    return HMC{AD, ()}(n_iters, epsilon, tau, 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int, space...) where AD
    return HMC{AD, space}(n_iters, epsilon, tau, 0)
end
function HMC{AD1}(alg::HMC{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    return HMC{AD1, space}(alg.n_iters, alg.epsilon, alg.tau, new_gid)
end
function HMC{AD, space}(alg::HMC, new_gid::Int) where {AD, space}
    return HMC{AD, space}(alg.n_iters, alg.epsilon, alg.tau, new_gid)
end

Sampler(alg::Hamiltonian) =  Sampler(alg, STAN_DEFAULT_ADAPT_CONF::DEFAULT_ADAPT_CONF_TYPE)
Sampler(alg::Hamiltonian, adapt_conf::DEFAULT_ADAPT_CONF_TYPE) = begin
    info=Dict{Symbol, Any}()

    # For state infomation
    info[:lf_num] = 0
    info[:eval_num] = 0

    # Adapt configuration
    info[:adapt_conf] = adapt_conf

    Sampler(alg, info)
end
