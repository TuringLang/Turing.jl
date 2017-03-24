include("support/hmc_helper.jl")
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
  n_samples:: Int       # number of samples
  lf_size  :: Float64   # leapfrog step size
  lf_num   :: Int       # leapfrog step number
  space    :: Set       # sampling space, emtpy means all
  function HMC(lf_size::Float64, lf_num::Int, space...)
    HMC(1, lf_size, lf_num, space...)
  end
  function HMC(n_samples, lf_size, lf_num)
    new(n_samples, lf_size, lf_num, Set())
  end
  function HMC(n_samples, lf_size, lf_num, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_samples, lf_size, lf_num, space)
  end
end

type HMCSampler{HMC} <: GradientSampler{HMC}
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

function step(model, data, spl::Sampler{HMC}, varInfo::VarInfo, is_first::Bool)
  if is_first
    # Run the model for the first time
    dprintln(2, "initialising...")
    varInfo = model(data, varInfo, spl)
    # Return
    true, varInfo
  else
    # Set parameters
    ϵ, τ = spl.alg.lf_size, spl.alg.lf_num

    dprintln(2, "sampling momentum...")
    p = Dict(k => randn(length(varInfo[k])) for k in keys(varInfo))

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, data, varInfo, spl)

    dprintln(3, "first gradient...")
    val∇E = gradient(varInfo, model, data, spl)

    dprintln(2, "leapfrog stepping...")
    for t in 1:τ  # do 'leapfrog' for each var
      varInfo, val∇E, p = leapfrog(varInfo, val∇E, p, ϵ, model, data, spl)
    end

    dprintln(2, "computing new H...")
    H = find_H(p, model, data, varInfo, spl)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)      # accepted
      true, varInfo
    else                                # rejected
      false, varInfo
    end
  end
end

function Base.run(model::Function, data::Dict, spl::Sampler{HMC})
  # initialization
  n =  spl.alg.n_samples
  task = current_task()
  t_start = time()  # record the start time of HMC
  accept_num = 0    # record the accept number
  varInfo = VarInfo()

  # HMC steps
  for i = 1:n
    dprintln(2, "recording old θ...")
    old_vals = deepcopy(varInfo.vals)
    dprintln(2, "HMC stepping...")
    is_accept, varInfo = step(model, data, spl, varInfo, i==1)
    if is_accept    # accepted => store the new predcits
      spl.samples[i].value = deepcopy(task.storage[:turing_predicts])
      accept_num = accept_num + 1
    else            # rejected => store the previous predcits
      varInfo.vals = old_vals
      spl.samples[i] = spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function assume(spl::Union{Void, HMCSampler{HMC}}, dist::Distribution, uid::String, sym::Symbol, vi::VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  # ct = current_trace()
  # r = rand(ct, dist)
  local r
  if ~haskey(vi, uid)
    dprintln(2, "sampling prior...")
    r = rand(dist)
    val = vectorize(dist, link(dist, r))      # X -> R and vectorize
    addvar!(vi, uid, val, sym, dist)
    push!(vi.randomness, r) # sync randomness and vals
  else
    dprintln(2, "fetching vals...")
    val = vi[uid]
    r = invlink(dist, reconstruct(dist, val)) # R -> X and reconstruct
    sync(vi, uid, r)        # sync R and X
  end
  if spl == nothing || isempty(spl.alg.space) || sym in spl.alg.space
    nothing
  else
    nothing
  end
  vi.logjoint += logpdf(dist, r, true)
  r
  # if ~haskey(vi, uid)   # first time -> generate
  #   # Sample a new prior
  #   dprintln(2, "sampling prior...")
  #   r = rand(dist)
  #
  #   # Transform
  #   val = vectorize(dist, link(dist, r)) # X -> R and vectorize
  #
  #   # Store the generated uid if it's in space
  #   if spl == nothing || isempty(spl.alg.space) || sym in spl.alg.space
  #     addvar!(vi, uid, val, sym, dist)
  #   end
  # else                              # not first time -> replay
  #   # Replay varibale
  #   dprintln(2, "fetching vals...")
  #   val = vi[uid]
  # end
  #
  # # Step 2 - Reconstruct variable
  # dprintln(2, "reconstructing vals...")
  # val = reconstruct(dist, val)  # reconstruct
  # val = invlink(dist, val)      # R -> X
  #
  # # Computing logjoint
  # dprintln(2, "computing logjoint...")
  # vi.logjoint += logpdf(dist, val, true)
  # dprintln(2, "compute logjoint done")
  # dprintln(2, "assume done")
  # return val
end

function observe(spl::Union{Void, HMCSampler{HMC}}, d::Distribution, value, vi::VarInfo)
  dprintln(2, "observing...")
  if length(value) == 1
    vi.logjoint += logpdf(d, Dual(value))
  else
    vi.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

function sample(model::Function, data::Dict, alg::HMC, chunk_size::Int)
  global CHUNKSIZE = chunk_size;
  sampler = HMCSampler{HMC}(alg);
  run(model, data, sampler)
end

function sample(model::Function, alg::HMC, chunk_size::Int)
  global CHUNKSIZE = chunk_size;
  sampler = HMCSampler{HMC}(alg);
  run(model, Dict(), sampler)
end

function sample(model::Function, data::Dict, alg::HMC)
  sampler = HMCSampler{HMC}(alg);
  run(model, data, sampler)
end

function sample(model::Function, alg::HMC)
  sampler = HMCSampler{HMC}(alg);
  run(model, Dict(), sampler)
end
