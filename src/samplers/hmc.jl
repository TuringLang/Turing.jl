include("support/hmc_core.jl")

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

type HMCSampler{HMC} <: Sampler{HMC}
  alg        ::HMC                          # the HMC algorithm info
  samples    ::Array{Sample}                # samples
  function HMCSampler(alg::HMC)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    new(alg, samples)
  end
end

function step(model, spl::Sampler{HMC}, varInfo::VarInfo, is_first::Bool)
  if is_first
    # Run the model for the first time
    dprintln(2, "initialising...")
    varInfo = runmodel(model, varInfo, spl)
    # Return
    true, varInfo
  else
    # Set parameters
    ϵ, τ = spl.alg.lf_size, spl.alg.lf_num

    dprintln(2, "sampling momentum...")
    p = Dict(uid(k) => randn(length(varInfo[k])) for k in keys(varInfo))
    if ~isempty(spl.alg.space)
      p = filter((k, p) -> getsym(varInfo, k) in spl.alg.space, p)
    end

    dprintln(3, "X -> R...")
    varInfo = link(varInfo, spl)

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, varInfo, spl)

    dprintln(3, "first gradient...")
    val∇E = gradient(varInfo, model, spl)

    dprintln(2, "leapfrog stepping...")
    for t in 1:τ  # do 'leapfrog' for each var
      varInfo, val∇E, p = leapfrog(varInfo, val∇E, p, ϵ, model, spl)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, model, varInfo, spl)

    dprintln(3, "R -> X...")
    varInfo = invlink(varInfo, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    realpart!(varInfo)

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)      # accepted
      true, varInfo
    else                                # rejected
      false, varInfo
    end
  end
end

setchunksize(chun_size::Int) = global CHUNKSIZE = chunk_size
sample(model::Function, alg::HMC) = sample(model, alg, CHUNKSIZE)

# NOTE: in the previous code, `sample` would call `run`; this is
# now simplified: `sample` and `run` are merged into one function.
function sample(model::Function, alg::HMC, chunk_size::Int)
  global CHUNKSIZE = chunk_size;
  global sampler = HMCSampler{HMC}(alg);

  spl = sampler
  # initialization
  n =  spl.alg.n_samples
  task = current_task()
  accept_num = 0    # record the accept number
  varInfo = VarInfo()

  # HMC steps
  @showprogress 1 "[HMC] Sampling..." for i = 1:n
    dprintln(2, "recording old θ...")
    old_vals = deepcopy(varInfo.vals)
    dprintln(2, "HMC stepping...")
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

function assume(spl::HMCSampler{HMC}, dist::Distribution, vn::VarName, vi::VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  r = rand(vi, vn, dist, spl)
  vi.logjoint += logpdf(dist, r, istransformed(vi, vn))
  r
end

# NOTE: TRY TO REMOVE Void through defining a special type for gradient based algs.
function observe(spl::Union{Void, HMCSampler{HMC}}, d::Distribution, value, vi::VarInfo)
  dprintln(2, "observing...")
  if length(value) == 1
    vi.logjoint += logpdf(d, Dual(value))
  else
    vi.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

rand(vi::VarInfo, vn::VarName, dist::Distribution, spl::HMCSampler{HMC}) = isempty(spl.alg.space) || vn.sym in spl.alg.space ? randr(vi, vn, dist, spl.alg.group_id, spl, false) : randr(vi, vn, dist)
