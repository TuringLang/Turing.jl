
"""
    PG(n_particles::Int, n_iters::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
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

sample(gdemo([1.5, 2]), PG(100, 100))
```
"""
mutable struct PG{space, F} <: AbstractGibbs
    n_particles           ::    Int         # number of particles used
    n_iters               ::    Int         # number of iterations
    resampler             ::    F           # function to resample
    gid                   ::    Int         # group ID
end
PG(n1::Int, n2::Int) = PG{(), typeof(resample_systematic)}(n1, n2, resample_systematic, 0)
function PG(n1::Int, n2::Int, space...)
    F = typeof(resample_systematic)
    PG{space, F}(n1, n2, resample_systematic, 0)
end
function PG(alg::PG{space, F}, new_gid::Int) where {space, F}
    return PG{space, F}(alg.n_particles, alg.n_iters, alg.resampler, new_gid)
end
PG{space, F}(alg::PG, new_gid::Int) where {space, F} = PG{space, F}(alg.n_particles, alg.n_iters, alg.resampler, new_gid)

const CSMC = PG # type alias of PG as Conditional SMC

function Sampler(alg::PG)
    info = Dict{Symbol, Any}()
    info[:logevidence] = []
    Sampler(alg, info)
end
