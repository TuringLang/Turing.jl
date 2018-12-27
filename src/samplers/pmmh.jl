"""
    PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})

Particle independant Metropolis–Hastings and
Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

- `n_iters::Int` : Number of iterations to run.
- `smc_alg:::SMC` : An [`SMC`](@ref) algorithm to use.
- `parameters_algs::Tuple{MH}` : An [`MH`](@ref) algorithm, which includes a
sample space specification.
"""
mutable struct PMMH{space, A<:Tuple} <: InferenceAlgorithm
    n_iters               ::    Int               # number of iterations
    algs                  ::    A                 # Proposals for state & parameters
    gid                   ::    Int               # group ID
end
function PMMH(n_iters::Int, smc_alg::SMC, parameter_algs...)
    algs = tuple(parameter_algs..., smc_alg)
    PMMH{buildspace(algs)}(n_iters, algs, 0)
end
PMMH(alg::PMMH, new_gid) = PMMH{getspace(alg)}(alg.n_iters, alg.algs, new_gid)

function PIMH(n_iters::Int, smc_alg::SMC)
    algs = tuple(smc_alg)
    PMMH{buildspace(algs)}(n_iters, algs, 0)
end

@inline function get_pmmh_samplers(subalgs, model, n, alg, alg_str)
  if length(subalgs) == 0
      return ()
  else
      subalg = subalgs[1]
      if typeof(subalg) == MH && subalg.n_iters != 1
          warn("[$alg_str] number of iterations greater than 1 is useless for MH since it is only used for its proposal")
      end
      if isa(subalg, Union{SMC, MH})
          return (Sampler(typeof(subalg)(subalg, n + 1 - length(subalgs)), model), get_pmmh_samplers(Base.tail(subalgs), model, n, alg, alg_str)...)
      else
          error("[$alg_str] unsupport base sampling algorithm $alg")
      end
  end
end

function Sampler(alg::PMMH, model::Model)
    alg_str = "PMMH"
    n_samplers = length(alg.algs)
    samplers = get_pmmh_samplers(alg.algs, model, n_samplers, alg, alg_str)
    verifyspace(alg.algs, model.pvars, alg_str)
    info = Dict{Symbol, Any}()
    info[:old_likelihood_estimate] = -Inf # Force to accept first proposal
    info[:old_prior_prob] = 0.0
    info[:samplers] = samplers

    Sampler(alg, info)
end
