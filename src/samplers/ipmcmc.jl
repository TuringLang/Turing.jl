
"""
    IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

- `n_particles::Int` : Number of particles to use.
- `n_iters::Int` : Number of iterations to employ.
- `n_nodes::Int` : The number of nodes running SMC and CSMC.
- `n_csmc_nodes::Int` : The number of CSMC nodes.

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), IPMCMC(100, 100, 4, 2))
```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).
"""
mutable struct IPMCMC{space, F} <: InferenceAlgorithm
    n_particles           ::    Int         # number of particles used
    n_iters               ::    Int         # number of iterations
    n_nodes               ::    Int         # number of nodes running SMC and CSMC
    n_csmc_nodes          ::    Int         # number of nodes CSMC
    resampler             ::    F           # function to resample
    gid                   ::    Int         # group ID
end
function IPMCMC(n1::Int, n2::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, 32, 16, resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, n3, Int(ceil(n3/2)), resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, n3, n4, resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int, space...)
    F = typeof(resample_systematic)
    IPMCMC{space, F}(n1, n2, n3, n4, resample_systematic, 0)
end
function IPMCMC(alg::IPMCMC, new_gid::Int)
    F = typeof(alg.resampler)
    @unpack n_particles, n_iters, n_nodes, n_csmc_nodes, resampler = alg
    S = getspace(alg)
    return IPMCMC{S, F}(n_particles, n_iters, n_nodes, n_csmc_nodes, resampler, new_gid)
end

function Sampler(alg::IPMCMC)
    # Create SMC and CSMC nodes
    samplers = Array{Sampler}(undef, alg.n_nodes)
    # Use resampler_threshold=1.0 for SMC since adaptive resampling is invalid in this setting
    default_CSMC = CSMC(alg.n_particles, 1, alg.resampler, getspace(alg), 0)
    default_SMC = SMC(alg.n_particles, alg.resampler, 1.0, false, getspace(alg), 0)

    for i in 1:alg.n_csmc_nodes
      samplers[i] = Sampler(CSMC(default_CSMC, i))
    end
    for i in (alg.n_csmc_nodes+1):alg.n_nodes
      samplers[i] = Sampler(SMC(default_SMC, i))
    end

    info = Dict{Symbol, Any}()
    info[:samplers] = Tuple(samplers)

    Sampler(alg, info)
end
