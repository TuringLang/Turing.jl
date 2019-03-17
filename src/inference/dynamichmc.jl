struct DynamicNUTS{AD, space} <: Hamiltonian{AD, space}
    n_iters   ::  Integer   # number of samples
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
DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
function DynamicNUTS{AD}(n_iters::Integer, space...) where AD
    DynamicNUTS{AD, space}(n_iters, 0)
end

function _sample(vi, samples, spl, model, alg::DynamicNUTS, chunk_size=CHUNKSIZE[])
    if ADBACKEND[] == :forward_diff
        default_chunk_size = CHUNKSIZE[]  # record global chunk size
        setchunksize(chunk_size)        # set temp chunk size
    end
    
    if spl.alg.gid == 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function _lp(x)
        value, deriv = gradient_logp(x, vi, model, spl)
        return ValueGradient(value, deriv)
    end

    chn_dynamic, _ = NUTS_init_tune_mcmc(FunctionLogDensity(length(vi[spl]), _lp), alg.n_iters)

    for i = 1:alg.n_iters
        vi[spl] = chn_dynamic[i].q
        samples[i].value = Sample(vi, spl).value
    end

    return Chain(0.0, samples)
end
