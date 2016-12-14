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
end

type HMCSampler{HMC} <: GradientSampler{HMC}
  alg         :: HMC                          # the HMC algorithm info
  model       :: Function                     # model function
  values      :: GradientInfo                 # container for variables
  dists       :: Dict{VarInfo, Distribution}  # dictionary from variables to the corresponding distributions
  samples     :: Array{Sample}                # samples
  predicts    :: Dict{Symbol, Any}            # outputs
  first       :: Bool                         # the first run flag

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    predicts = Dict{Symbol, Any}()
    values = GradientInfo()   # GradientInfo initialize logjoint as Dual(0)
    dists = Dict{VarInfo, Distribution}()
    new(alg, model, values, dists, samples, predicts, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  # Half momentum step
  function half_momentum_step(p, val∇E)
    dprintln(5, "half_momentum_step...")
    for k in keys(p)
      p[k] -= ϵ * val∇E[k] / 2
    end
    return p
  end

  # Find logjoint
  # NOTE: it returns logjoint but not -logjoint
  function find_logjoint()
    consume(Task(spl.model))
    logjoint = spl.values.logjoint
    spl.values.logjoint = Dual(0)
    return logjoint
  end

  # Run the model for the first time
  dprintln(2, "initialising...")
  find_logjoint()
  spl.first = false

  # Store the first predicts
  spl.samples[1].value = deepcopy(spl.predicts)

  # Set parameters
  n = spl.alg.n_samples
  ϵ = spl.alg.lf_size
  τ = spl.alg.lf_num
  accept_num = 1        # the first samples is always accepted

  # Sampling
  for i = 2:n
    dprintln(3, "stepping...")

    # Initialization
    has_run = false
    oldH = 0
    H = 0

    # Record old state
    old_values = deepcopy(spl.values)

    # Generate random momentum
    p = Dict{Any, Any}()
    for k in keys(spl.values)
      p[k] = randn(length(spl.values[k]))
    end

    # Record old Hamiltonian
    dprintln(4, "old H...")
    for k in keys(p)
      oldH += p[k]' * p[k] / 2
    end
    oldH += realpart(-find_logjoint())

    # Get gradient dict
    dprintln(4, "first gradient...")
    val∇E = get_gradient_dict(spl.values, spl.model)

    # Do 'leapfrog' for each var
    dprintln(4, "leapfrog...")
    for t in 1:τ
      p = half_momentum_step(p, val∇E)  # half step for momentum
      for k in keys(spl.values)         # full step for state
        spl.values[k] = Array{Any}(spl.values[k] + ϵ * p[k])
      end
      val∇E = get_gradient_dict(spl.values, spl.model)
      p = half_momentum_step(p, val∇E)  # half step for momentum
    end

    # Claculate the new Hamiltonian
    dprintln(4, "new H...")
    for k in keys(p)
      H += p[k]' * p[k] / 2
    end
    H += realpart(-find_logjoint())

    # Calculate the difference in Hamiltonian
    ΔH = H - oldH
    ΔH = ΔH[1]  # Vector{Any, 1} -> Any

    # Decide wether to accept or not
    if ΔH < 0
      acc = true
    elseif rand() < exp(-ΔH)
      acc = true
    else
      acc = false
    end

    if ~acc # rejected => store the previous predcits
      spl.values = old_values
      spl.samples[i] = spl.samples[i - 1]
    else    # accepted => store the new predcits
      spl.samples[i].value = deepcopy(spl.predicts)
      accept_num += 1
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate)")
  return Chain(0, spl.samples)    # wrap the result by Chain
end

function assume(spl :: HMCSampler{HMC}, dist :: Distribution, var :: VarInfo)
  # Step 1 - Generate or replay variable
  dprintln(2, "assuming...")
  if spl.first  # first time -> generate
    # Build {var -> dist} dictionary
    spl.dists[var] = dist

    # Sample a new prior
    dprintln(2, "sampling prior...")
    r = rand(dist)

    # Transform
    v = link(dist, r)        # X -> R
    val = vectorize(dist, v) # vectorize

    # Store the generated var
    addVarInfo(spl.values, var, val)
  else         # not first time -> replay
    # Replay varibale
    dprintln(2, "fetching values...")
    val = spl.values[var]
  end

  # Step 2 - Reconstruct variable
  dprintln(2, "reconstructing values...")
  val = reconstruct(dist, val)  # reconstruct
  val = invlink(dist, val)      # R -> X

  # Computing logjoint
  dprintln(2, "computing logjoint...")
  spl.values.logjoint += logpdf(dist, val, true)
  dprintln(2, "compute logjoint done")
  dprintln(2, "assume done")
  return val
end

function observe(spl :: HMCSampler{HMC}, d :: Distribution, value)
  dprintln(2, "observing...")
  if length(value) == 1
    spl.values.logjoint += logpdf(d, Dual(value))
  else
    spl.values.logjoint += logpdf(d, map(x -> Dual(x), value))
  end
  dprintln(2, "observe done")
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  dprintln(2, "predicting...")
  spl.predicts[name] = realpart(value)
  dprintln(2, "predict done")
end

function sample(model :: Function, alg :: HMC)
  global sampler = HMCSampler{HMC}(alg, model);
  run(sampler)
end
