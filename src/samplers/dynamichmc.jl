struct DynamicNUTS{space} <: Hamiltonian
    n_iters   ::  Integer   # number of samples
    gid       ::  Integer   # group ID
end
DynamicNUTS(n_iters, gid) = DynamicNUTS{()}(n_iters, gid)

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
    DynamicNUTS{space}(n_iters, 0)
end

function Sampler(alg::DynamicNUTS{T}) where T <: Hamiltonian
    return Sampler(alg, Dict{Symbol,Any}())
end
