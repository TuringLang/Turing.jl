
doc"""
    IS(n_particles::Int)

Importance sampler.

Usage:

```julia
IS(1000)
```

Example:

```julia
@model example begin
  ...
end

sample(example, IS(1000))
```
"""
immutable IS <: InferenceAlgorithm
  n_samples   ::  Int
  IS(n) = new(n)
end

Sampler(alg::IS) = begin
  info = Dict{Symbol, Any}()
  Sampler(alg, info)
end

function sample(model::Function, alg::IS)
  spl = Sampler(alg);
  samples = Array{Sample}(alg.n_samples)

  n = spl.alg.n_samples
  for i = 1:n
    vi = model(vi=VarInfo(), sampler=spl)
    samples[i] = Sample(vi)
  end
  le = logsum(map(x->x[:lp], samples)) - log(n)
  chn = Chain(exp(le), samples)
  return chn
end

assume(spl::Sampler{IS}, d::Distribution, vn::VarName, vi::VarInfo) = rand(vi, vn, d, spl)

observe(spl::Sampler{IS}, d::Distribution, value, vi::VarInfo) = begin
  vi.logp   += logpdf(d, value)
end

rand(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler{IS}) = randr(vi, vn, dist)
