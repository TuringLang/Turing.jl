include("support/hmc_helper.jl")

doc"""
    HMC(n_samples::Int64, lf_size::Float64, lf_num::Int64)

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
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
  space     ::  Tuple     # sampling space
  function HMC(n_samples, lf_size, lf_num)
    new(n_samples, lf_size, lf_num, ())
  end
  function HMC(n_samples, lf_size, lf_num, space)
    new(n_samples, lf_size, lf_num, space)
  end
end

type HMCSampler{HMC} <: GradientSampler{HMC}
  alg         :: HMC                          # the HMC algorithm info
  samples     :: Array{Sample}                # samples
  predicts    :: Dict{Symbol, Any}            # outputs

  function HMCSampler(alg :: HMC)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    predicts = Dict{Symbol, Any}()
    new(alg, samples, predicts)
  end
end

function step(model, data, spl::Sampler{HMC}, varInfo::VarInfo, n::Int64, ϵ::Float64, τ::Int64, is_first::Bool)
  if is_first
    # Run the model for the first time
    dprintln(2, "initialising...")
    varInfo = model(data, varInfo, spl)

    # Return
    true, varInfo
  else
    dprintln(2, "HMC stepping...")

    dprintln(2, "recording old θ...")
    old_values = deepcopy(varInfo.values)

    dprintln(2, "sampling momentum...")
    p = Dict(k => randn(length(varInfo[k])) for k in keys(varInfo))

    dprintln(2, "recording old H...")
    oldH = find_H(p, model, data, varInfo, spl)

    dprintln(3, "first gradient...")
    val∇E = get_gradient_dict(varInfo, model, data, spl)

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
      varInfo.values = old_values
      false, varInfo
    end
  end
end

function Base.run(model, data, spl::Sampler{HMC})
  # Set parameters
  n, ϵ, τ = spl.alg.n_samples, spl.alg.lf_size, spl.alg.lf_num

  # initialization
  t_start = time()  # record the start time of HMC
  accept_num = 0    # record the accept number
  values = VarInfo()

  # HMC steps
  for i = 1:n
    is_accept, values = step(model, data, spl, values, n, ϵ, τ, i==1)
    if is_accept  # accepted => store the new predcits
      spl.samples[i].value = deepcopy(spl.predicts)
      accept_num = accept_num + 1
    else          # rejected => store the previous predcits
      spl.samples[i] = spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function assume(spl :: Union{Void, HMCSampler{HMC}}, dist :: Distribution, var :: Var, varInfo::VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  if ~haskey(varInfo.values, var)  # first time -> generate
    # Sample a new prior
    dprintln(2, "sampling prior...")
    r = rand(dist)

    # Transform
    v = link(dist, r)        # X -> R
    val = vectorize(dist, v) # vectorize

    # Store the generated var
    varInfo.values[var] = val
  else         # not first time -> replay
    # Replay varibale
    dprintln(2, "fetching values...")
    val = varInfo[var]
  end

  # Step 2 - Reconstruct variable
  dprintln(2, "reconstructing values...")
  val = reconstruct(dist, val)  # reconstruct
  val = invlink(dist, val)      # R -> X

  # Computing logjoint
  dprintln(2, "computing logjoint...")
  varInfo.logjoint += logpdf(dist, val, true)
  dprintln(2, "compute logjoint done")
  dprintln(2, "assume done")
  return val
end

function observe(spl :: Union{Void, HMCSampler{HMC}}, d :: Distribution, value, varInfo::VarInfo)
  dprintln(2, "observing...")
  if length(value) == 1
    varInfo.logjoint += logpdf(d, Dual(value))
  else
    varInfo.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  dprintln(2, "predicting...")
  spl.predicts[name] = realpart(value)
  dprintln(2, "predict done")
end

function sample(model::Function, data::Dict, alg::HMC, chunk_size::Int64)
  global CHUNKSIZE = chunk_size;
  sampler = HMCSampler{HMC}(alg);
  run(model, data, sampler)
end

function sample(model::Function, data::Dict, alg::HMC)
  sampler = HMCSampler{HMC}(alg);
  run(model, data, sampler)
end

function sample(model::Function, alg::HMC, chunk_size::Int64)
  global CHUNKSIZE = chunk_size;
  sampler = HMCSampler{HMC}(alg);
  run(model, Dict(), sampler)
end

function sample(model::Function, alg::HMC)
  sampler = HMCSampler{HMC}(alg);
  run(model, Dict(), sampler)
end
