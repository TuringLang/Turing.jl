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
  dists       :: Dict{VarInfo, Distribution}  # variable to its distribution
  samples     :: Array{Sample}                # samples
  predicts    :: Dict{Symbol, Any}            # outputs
  first       :: Bool                         # the first run flag

  function HMCSampler(alg :: HMC, model :: Function)
    values = GradientInfo()   # GradientInfo initialize logjoint as Dual(0)
    dists = Dict{VarInfo, Distribution}()
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    predicts = Dict{Symbol, Any}()
    first = true
    new(alg, model, values, dists, samples, predicts, first)
  end
end

function Base.run(spl :: Sampler{HMC})

  # Half momentum step
  function half_momentum_step(p, val∇E)
    dprintln(3, "half_momentum_step...")
    for k in keys(p)
      p[k] -= ϵ * val∇E[k] / 2
    end
    p
  end

  # Leapfrog step
  function leapfrog(values, p, model)
    # Get gradient dict
    dprintln(3, "first gradient...")
    val∇E = get_gradient_dict(values, model)

    # Do 'leapfrog' for each var
    dprintln(3, "leapfrog...")
    for t in 1:τ
      p = half_momentum_step(p, val∇E)  # half step for momentum
      for k in keys(values)             # full step for state
        values[k] = Array{Any}(values[k] + ϵ * p[k])
      end
      val∇E = get_gradient_dict(values, model)
      p = half_momentum_step(p, val∇E)  # half step for momentum
    end

    # Return updated θ and momentum
    values, p
  end

  # Find logjoint
  # NOTE: it returns logjoint but not -logjoint
  function find_logjoint(model, values)
    consume(Task(model))        # run model
    logjoint = values.logjoint  # get logjoint
    values.logjoint = Dual(0)   # reset logjoint
    logjoint
  end

  # Compute Hamiltonian
  function find_H(p, model, values)
    H = 0
    for k in keys(p)
      H += p[k]' * p[k] / 2
    end
    H += realpart(-find_logjoint(model, values))
    H[1]  # Vector{Any, 1} -> Any
  end

  t_start = time()  # record the start time of HMC

  # Run the model for the first time
  dprintln(2, "initialising...")
  find_logjoint(spl.model, spl.values)
  spl.first = false

  # Store the first predicts
  spl.samples[1].value = deepcopy(spl.predicts)

  # Set parameters
  n, ϵ, τ = spl.alg.n_samples, spl.alg.lf_size, spl.alg.lf_num
  accept_num = 1        # the first samples is always accepted

  # HMC steps
  for i = 2:n
    dprintln(2, "HMC stepping...")

    dprintln(2, "recording old θ...")
    old_values = deepcopy(spl.values)

    dprintln(2, "sampling momentum...")
    p = Dict{Any, Any}()
    for k in keys(spl.values)
      p[k] = randn(length(spl.values[k]))
    end

    dprintln(2, "recording old H...")
    oldH = find_H(p, spl.model, spl.values)

    dprintln(2, "leapfrog stepping...")
    spl.values, p = leapfrog(spl.values, p, spl.model)

    dprintln(2, "computing new H...")
    H = find_H(p, spl.model, spl.values)

    dprintln(2, "computing ΔH...")
    ΔH = H - oldH

    dprintln(2, "decide wether to accept...")
    if ΔH < 0 || rand() < exp(-ΔH)  # accepted => store the new predcits
      spl.samples[i].value, accept_num = deepcopy(spl.predicts), accept_num + 1
    else                            # rejected => store the previous predcits
      spl.values, spl.samples[i] = old_values, spl.samples[i - 1]
    end
  end

  accept_rate = accept_num / n    # calculate the accept rate
  println("[HMC]: Finshed with accept rate = $(accept_rate) within $(time() - t_start) seconds")
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
