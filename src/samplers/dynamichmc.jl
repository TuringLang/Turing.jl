struct DynamicNUTS{T} <: Hamiltonian
    n_iters   ::  Integer   # number of samples
    space     ::  Set{T}    # sampling space, emtpy means all
    gid       ::  Integer   # group ID
end

"""
    DynamicNUTS(n_iters::Integer)

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.
To use it, make sure you have the DynamicHMC package installed.

```julia
# Import Turing and DynamicHMC.
using DynamicHMC, Turing

# Model definition.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

# Pull 2,000 samples using DynamicNUTS.
chn = sample(gdemo(1.5, 2.0), DynamicNUTS(2000))
```
"""
function DynamicNUTS(n_iters::Integer, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    DynamicNUTS(n_iters, _space, 0)
end

function Sampler(alg::DynamicNUTS{T}) where T <: Hamiltonian
  return Sampler(alg, Dict{Symbol,Any}())
end

function sample(model::Function, alg::DynamicNUTS, chunk_size=CHUNKSIZE[]) where T <: Hamiltonian
    if ADBACKEND[] == :forward_diff
        default_chunk_size = CHUNKSIZE[]  # record global chunk size
        setchunksize(chunk_size)        # set temp chunk size
    end

    spl = Sampler(alg)

    n = alg.n_iters
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    vi = VarInfo()
    Base.invokelatest(model, vi, HamiltonianRobustInit())

    if spl.alg.gid == 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function _lp(x)
        value, deriv = gradient(x, vi, model, spl)
        return ValueGradient(-value, -deriv)
    end

    chn_dynamic, _ = NUTS_init_tune_mcmc(FunctionLogDensity(length(vi[spl]), _lp), alg.n_iters)

    for i = 1:alg.n_iters
        vi[spl] = chn_dynamic[i].q
        samples[i].value = Sample(vi, spl).value
    end

    if ADBACKEND[] == :forward_diff
        setchunksize(default_chunk_size)      # revert global chunk size
    end

    return Chain(0, samples)
end
