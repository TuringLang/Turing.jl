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
  alg         :: HMC
  model       :: Function
  values      :: GradientInfo
  dists       :: Dict{VarInfo, Distribution}
  samples     :: Array{Sample}
  predicts    :: Dict{Symbol, Any}
  first       :: Bool

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    predicts = Dict{Symbol, Any}()
    values = GradientInfo()   # GradientInfo Initialize logjoint as Dual(0)
    dists = Dict{VarInfo, Distribution}()
    new(alg, model, values, dists, samples, predicts, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  # Function to make half momentum step
  function half_momentum_step(p, val∇E)
    dprintln(5, "half_momentum_step...")
    for k in keys(p)
      p[k] -= ϵ * val∇E[k] / 2
    end
    return p
  end
  # Run the model for the first time
  dprintln(2, "initialising...")
  consume(Task(spl.model))
  spl.values.logjoint = Dual(0)
  spl.first = false
  # Store the first predicts
  spl.samples[1].value = deepcopy(spl.predicts)
  n = spl.alg.n_samples
  ϵ = spl.alg.lf_size
  τ = spl.alg.lf_num
  accept_num = 1
  # Sampling
  for i = 2:n
    has_run = false
    oldH = 0
    H = 0
    # Record old state
    old_values = deepcopy(spl.values)
    # Run the step until successful
    while has_run == false
      dprintln(3, "stepping...")
      # Assume the step is successful
      has_run = true
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
      consume(Task(spl.model))
      oldH += realpart(-spl.values.logjoint)
      spl.values.logjoint = Dual(0)
      # Get gradient dict
      dprintln(4, "first gradient...")
      val∇E = get_gradient_dict(spl.values, spl.model)
      dprintln(4, "leapfrog...")
      # 'leapfrog' for each var
      for t in 1:τ
        p = half_momentum_step(p, val∇E)
        # Make a full step for state
        for k in keys(spl.values)
          # X -> R and move
          dprintln(5, "reconstruct...")
          real = reconstruct(spl.dists[k], spl.values[k])
          dprintln(5, "X -> R...")
          real = link(spl.dists[k], real)
          dprintln(5, "move...")
          real += ϵ * reconstruct(spl.dists[k], p[k])
          real = length(real) == 1 ? real[1] : real       # Array{T}[1] → T for invlink()
          # R -> X and store
          dprintln(5, "R -> X...")
          spl.values[k] = vectorize(spl.dists[k], invlink(spl.dists[k], real))
          # spl.values[k] = Array{Any}(spl.values[k] + ϵ * p[k])
        end
        val∇E = get_gradient_dict(spl.values, spl.model)
        p = half_momentum_step(p, val∇E)
      end
      # Claculate the new Hamiltonian
      dprintln(4, "new H...")
      for k in keys(p)
        H += p[k]' * p[k] / 2
      end
      consume(Task(spl.model))
      H += realpart(-spl.values.logjoint)
      spl.values.logjoint = Dual(0)
    end
    # Calculate the difference in Hamiltonian
    ΔH = H - oldH
    # Vector{Any, 1} -> Any
    ΔH = ΔH[1]
    # Decide wether to accept or not
    if ΔH < 0
      acc = true
    elseif rand() < exp(-ΔH)
      acc = true
    else
      acc = false
    end
    # Rewind of rejected
    if ~acc
      spl.values = old_values
      # Store the previous predcits
      spl.samples[i] = spl.samples[i - 1]
    else
      # Store the new predcits
      spl.samples[i].value = deepcopy(spl.predicts)
      accept_num += 1
    end
  end
  # Wrap the result by Chain
  results = Chain(0, spl.samples)
  accept_rate = accept_num / n
  println("[HMC]: Finshed with accept rate = $(accept_rate)")
  return results
end

function assume(spl :: HMCSampler{HMC}, d :: Distribution, var :: VarInfo)
  dprintln(2, "assuming...")
  # 1. Gen values and vectorize if necessary

  # TODO: Change the first running condition
  # If it's the first time running the program

  if spl.first
    # Record dist
    spl.dists[var] = d

    # Generate a new var
    dprintln(2, "generating values...")
    r = rand(d)
    val = r
    val = vectorize(d, r)
    # Store the generated var
    addVarInfo(spl.values, var, val)
  # If not the first time
  else
    # Fetch the existing var
    dprintln(2, "fetching values...")
    val = spl.values[var]
  end

  # 2. reconstruct values
  dprintln(2, "reconstructing values...")
  dim = size(spl.dists[var])
  val = reconstruct(d, val)

  dprintln(2, "computing logjoint...")
  spl.values.logjoint += logpdf(d, val)
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

sample(model :: Function, alg :: HMC) = (
                                        global sampler = HMCSampler{HMC}(alg, model);
                                        run(sampler)
                                        )
