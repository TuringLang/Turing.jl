if debug_level == 0
  RerunThreshold = 250
else
  RerunThreshold = 1
end

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
  priors      :: GradientInfo
  samples     :: Array{Sample}
  logjoint    :: LogJoint
  predicts    :: Dict{Symbol, Any}
  first       :: Bool

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    logjoint = LogJoint(Dual(0))
    predicts = Dict{Symbol, Any}()
    priors = GradientInfo()
    new(alg, model, priors, samples, logjoint, predicts, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  # Function to make half momentum step
  function half_momentum_step(p, val∇E)
    for k in keys(p)
      p[k] -= ϵ * val∇E[k] / 2
    end
    return p
  end
  # Run the model for the first time
  dprintln(2, "initialising...")
  consume(Task(spl.model))
  spl.logjoint.val = Dual(0)
  spl.first = false
  rerun_num = 0
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
    old_priors = deepcopy(spl.priors)
    # Run the step until successful
    while has_run == false
      dprintln(3, "stepping...")
      # Assume the step is successful
      has_run = true
      try
        # Generate random momentum
        p = Dict{Any, Any}()
        for k in keys(spl.priors)
          p[k] = randn(length(spl.priors[k]))
        end
        # Record old Hamiltonian
        dprintln(4, "old H...")
        for k in keys(p)
          oldH += p[k]' * p[k] / 2
        end
        consume(Task(spl.model))
        oldH += realpart(-spl.logjoint.val)
        spl.logjoint.val = Dual(0)
        # Get gradient dict
        dprintln(4, "first gradient...")
        val∇E = get_gradient_dict(spl.priors, spl.model, spl.logjoint)
        dprintln(4, "leapfrog...")
        # 'leapfrog' for each prior
        for t in 1:τ
          p = half_momentum_step(p, val∇E)
          # Make a full step for state
          for k in keys(spl.priors)
            spl.priors[k] = Array{Any}(spl.priors[k] + ϵ * p[k])
          end
          val∇E = get_gradient_dict(spl.priors, spl.model, spl.logjoint)
          p = half_momentum_step(p, val∇E)
        end
        # Claculate the new Hamiltonian
        dprintln(4, "new H...")
        for k in keys(p)
          H += p[k]' * p[k] / 2
        end
        consume(Task(spl.model))
        H += realpart(-spl.logjoint.val)
        spl.logjoint.val = Dual(0)
      catch e
        # output error type
        dprintln(2, e)
        # Count re-run number
        rerun_num += 1
        # Only rerun for a threshold of times
        if rerun_num <= RerunThreshold
          # Revert the priors
          spl.priors = deepcopy(old_priors)
          # Set the model un-run parameters
          has_run = false
          oldH = 0
          H = 0
        else
          throw(BadParamError())
        end
      end
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
      spl.priors = old_priors
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
  println("[HMC]: Finshed with accept rate = $(accept_rate) (re-runs for $(rerun_num) times)")
  return results
end

function assume(spl :: HMCSampler{HMC}, d :: Distribution, prior :: Prior)
  dprintln(2, "assuming...")
  # TODO: Change the first running condition
  # If it's the first time running the program
  if spl.first
    # Generate a new prior
    r = rand(d)
    dim = length(r)
    if dim == 1
      val = Vector{Any}([Dual(r)])
    else
      val = Vector{Any}(map(x -> Dual(x), r))
    end
    # Store the generated prior
    addPrior(spl.priors, prior, val)
  # If not the first time
  else
    # Fetch the existing prior
    val = spl.priors[prior]
  end
  if isa(val, Array)
    if length(val) == 1
      # Turn Array{Any} to Any if necessary (this is due to randn())
      val = val[1]
    else
      # Turn Array{Any} to Array{T} if necessary (this is due to an update in Distributions.jl)
      T = typeof(val[1])
      val = Vector{T}(val)
    end
  end

  dprintln(2, "computing logjoint...")
  spl.logjoint.val += logpdf(d, val)
  dprintln(2, "compute logjoint done")
  dprintln(2, "assume done")
  return val
end

function observe(spl :: HMCSampler{HMC}, d :: Distribution, value)
  dprintln(2, "observing...")
  if length(value) == 1
    spl.logjoint.val += logpdf(d, Dual(value))
  else
    spl.logjoint.val += logpdf(d, map(x -> Dual(x), value))
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



# Error
type BadParamError <: Exception
end

Base.showerror(io::IO, e::BadParamError) = print(io, "HMC sampler terminates because of too many re-runs resulted from DomainError (over $(RerunThreshold)). This may be due to large value of ϵ and τ. Please try tuning these parameters.");
