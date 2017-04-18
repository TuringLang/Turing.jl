doc"""
    HMC(n_samples::Int, lf_size::Float64, lf_num::Int)

Hamiltonian Monte Carlo sampler.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
@model example begin
  ...
end

sample(example, HMC(1000, 0.05, 10))
```
"""
immutable HMC <: InferenceAlgorithm
  n_samples ::  Int       # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int       # leapfrog step number
  space     ::  Set       # sampling space, emtpy means all
  group_id  ::  Int
  function HMC(lf_size::Float64, lf_num::Int, space...)
    HMC(1, lf_size, lf_num, space..., 0)
  end
  function HMC(n_samples, lf_size, lf_num)
    new(n_samples, lf_size, lf_num, Set(), 0)
  end
  function HMC(n_samples, lf_size, lf_num, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_samples, lf_size, lf_num, space, 0)
  end
  HMC(alg::HMC, new_group_id::Int) = new(alg.n_samples, alg.lf_size, alg.lf_num, alg.space, new_group_id)
end

type HMCSampler{T} <: Sampler{T}
  alg     ::  T                         # the HMC algorithm info
  samples ::  Array{Sample}             # samples
  info    ::  Dict{Symbol, Any}         # sampler infomation
  function HMCSampler(alg::T)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    info = Dict{Symbol, Any}()
    new(alg, samples, info)
  end
end

function step(model, spl::Sampler{HMC}, vi::VarInfo, is_first::Bool)
  if is_first
    true, vi
  else
    # Set parameters
    ϵ, τ = spl.alg.lf_size, spl.alg.lf_num

    p = sample_momentum(vi, spl)

    dprintln(3, "X -> R...")
    vi = link(vi, spl)

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, vi, spl)

    dprintln(3, "first gradient...")
    grad = gradient(vi, model, spl)

    dprintln(2, "leapfrog stepping...")
    for t in 1:τ  # do 'leapfrog' for each var
      vi, grad, p = leapfrog(vi, grad, p, ϵ, model, spl)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, model, vi, spl)

    dprintln(3, "R -> X...")
    vi = invlink(vi, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    realpart!(vi)

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)      # accepted
      true, vi
    else                                # rejected
      false, vi
    end
  end
end

sample(model::Function, alg::Union{HMC, HMCDA}) = sample(model, alg, CHUNKSIZE)

# NOTE: in the previous code, `sample` would call `run`; this is
# now simplified: `sample` and `run` are merged into one function.
function sample(model::Function, alg::Union{HMC, HMCDA}, chunk_size::Int)
  global CHUNKSIZE = chunk_size;
  global sampler = HMCSampler{typeof(alg)}(alg);
  alg_str = isa(alg, HMC) ? "HMC" : "HMCDA"

  spl = sampler
  # initialization
  n =  spl.alg.n_samples
  task = current_task()
  accept_num = 0    # record the accept number
  varInfo = model()

  # HMC steps
  @showprogress 1 "[$alg_str] Sampling..." for i = 1:n
    dprintln(2, "recording old θ...")
    old_vals = deepcopy(varInfo.vals)
    dprintln(2, "$alg_str stepping...")
    is_accept, varInfo = step(model, spl, varInfo, i==1)
    if is_accept    # accepted => store the new predcits
      spl.samples[i].value = Sample(varInfo).value
      accept_num = accept_num + 1
    else            # rejected => store the previous predcits
      varInfo.vals = old_vals
      spl.samples[i] = spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate

  Chain(0, spl.samples)    # wrap the result by Chain
end

function assume{T<:Union{HMC,HMCDA}}(spl::HMCSampler{T}, dist::Distribution, vn::VarName, vi::VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  r = rand(vi, vn, dist, spl)
  vi.logjoint += logpdf(dist, r, istransformed(vi, vn))
  r
end

# NOTE: TRY TO REMOVE Void through defining a special type for gradient based algs.
function observe{T<:Union{HMC,HMCDA}}(spl::HMCSampler{T}, d::Distribution, value, vi::VarInfo)
  dprintln(2, "observing...")
  if length(value) == 1
    vi.logjoint += logpdf(d, Dual(value))
  else
    vi.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

rand{T<:Union{HMC,HMCDA}}(vi::VarInfo, vn::VarName, dist::Distribution, spl::HMCSampler{T}) = begin
  isempty(spl.alg.space) || vn.sym in spl.alg.space ?
    randr(vi, vn, dist, spl, false) :
    randr(vi, vn, dist)
end
