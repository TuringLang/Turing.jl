
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

type ImportanceSampler{IS} <: Sampler{IS}
  alg         ::  IS
  samples     ::  Vector{Sample}
  function ImportanceSampler(alg::IS)
    samples = Array{Sample}(alg.n_samples)
    new(alg, samples)
  end
end

function sample(model::Function, alg::IS)
  global sampler = ImportanceSampler{IS}(alg);
  spl = sampler

  n = spl.alg.n_samples
  for i = 1:n
    vi = model()
    spl.samples[i] = Sample(vi)
  end
  le = sum(map(x->x[:lp], spl.samples)) - log(n)
  chn = Chain(exp(le), spl.samples)
  return chn
end

assume(spl::ImportanceSampler{IS}, d::Distribution, vn::VarName, vi::VarInfo) = begin
  rand(d)
end

function observe(spl::ImportanceSampler{IS}, d::Distribution, value, vi::VarInfo)
  vi.logjoint   += logpdf(d, value)
end
