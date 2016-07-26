RerunThreshold = 250

immutable HMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
end

type HMCSampler{HMC} <: Sampler{HMC}
  alg         :: HMC
  model       :: Function
  samples     :: Array{Sample}
  logjoint    :: Dual{Float64}
  predicts    :: Dict{Symbol, Any}
  priors      :: Dict{Any, Any}
  first       :: Bool

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    logjoint = Dual(0, 0)
    predicts = Dict{Symbol, Any}()
    priors = Dict{Any, Any}()
    new(alg, model, samples, logjoint, predicts, priors, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  # Function to generate the gradient dictionary
  function get_gradient_dict()
    val∇E = Dict{Any, Any}()
    for k in keys(spl.priors)
      # TODO: Unify Float64 and Vector by making them all Vector
      if length(spl.priors[k]) == 1
        real = isa(spl.priors[k], Dual) ? realpart(spl.priors[k]) : realpart(spl.priors[k][1])
        # Set the dual part of the variable we want to find graident to 1
        spl.priors[k] = Dual(real, 1)
        # Run the model
        consume(Task(spl.model))
        # Reset dual part
        spl.priors[k] = Dual(real, 0)
        # Record graident
        val∇E[k] = -dualpart(spl.logjoint)
        # Reset the log joint
        spl.logjoint = Dual(0)
      else
        l = length(spl.priors[k])
        # To store the gradient vector
        g = zeros(l)
        for i = 1:l
          real = realpart(spl.priors[k][i])
          # Set the dual part of dimension i to 1
          spl.priors[k][i] = Dual(real, 1)
          # Run the model
          consume(Task(spl.model))
          # Reset
          spl.priors[k][i] = Dual(real, 0)
          # Record gradient of current dimension
          g[i] = dualpart(-spl.logjoint)
          # Reset the log joint
          spl.logjoint = Dual(0)
        end
        # Record gradient
        val∇E[k] = deepcopy(g)
      end
    end
    return val∇E
  end
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
  spl.logjoint = Dual(0)
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
        # Get gradient dict
        val∇E = get_gradient_dict()
        # Generate random momentum
        p = Dict{Any, Any}()
        for k in keys(spl.priors)
          p[k] = randn(length(spl.priors[k]))
        end
        # Record old Hamiltonian
        for k in keys(p)
          oldH += p[k]' * p[k] / 2
        end
        consume(Task(spl.model))
        oldH += realpart(-spl.logjoint)
        spl.logjoint = Dual(0)
        # 'leapfrog' for each prior
        for t in 1:τ
          p = half_momentum_step(p, val∇E)
          # Make a full step for state
          for k in keys(spl.priors)
            spl.priors[k] = forceVector(spl.priors[k], Dual{Real})
            spl.priors[k] += ϵ * p[k]
          end
          val∇E = get_gradient_dict()
          p = half_momentum_step(p, val∇E)
        end
        # Claculate the new Hamiltonian
        for k in keys(p)
          H += p[k]' * p[k] / 2
        end
        consume(Task(spl.model))
        H += realpart(-spl.logjoint)
        spl.logjoint = Dual(0)
      catch e
        # Revert the priors
        spl.priors = old_priors
        # Count re-run number
        rerun_num += 1
        # Only rerun for a threshold of times
        if rerun_num <= RerunThreshold
          # log
          # println("[HMC]: Error occurs in the 'leapfrog' step, re-running ($(rerun_num)).")
          # Set the model un-run
          has_run = false
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
  # TODO: Calculate the logevidence
  results = Chain(0, spl.samples)
  accept_rate = accept_num / n
  println("[HMC]: Finshed with accept rate = $(accept_rate) (re-runs for $(rerun_num) times)")
  return results
end

# TODO: Use another way to achieve replay. The current method fails when fetching arrays
function assume(spl :: HMCSampler{HMC}, dd :: dDistribution, p :: Prior)
  dprintln(2, "assuming...")
  name = string(p)
  # If it's the first time running the program
  if spl.first
    # Generate a new prior
    r = rand(dd)
    if length(r) == 1
      prior = Dual(r)
    else
      prior = Vector{Dual}(r)
    end
    # Store the generated prior
    spl.priors[name] = prior
  # If not the first time
  else
    # Fetch the existing prior
    prior = spl.priors[name]
  end
  # Turn Array{Any} to Any if necessary (this is due to randn())
  prior = (isa(prior, Array) && (length(prior) == 1)) ? prior[1] : prior
  spl.logjoint += log(pdf(dd, prior))
  return prior
end

function observe(spl :: HMCSampler{HMC}, dd :: dDistribution, value)
  dprintln(2, "observing...")
  if length(value) == 1
    spl.logjoint += log(pdf(dd, Dual(value)))
  else
    spl.logjoint += log(pdf(dd, Vector{Dual}(value)))
  end
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  dprintln(2, "predicting...")
  spl.predicts[name] = realpart(value)
end

sample(model :: Function, alg :: HMC) = (
                                        global sampler = HMCSampler{HMC}(alg, model);
                                        run(sampler)
                                        )



# Error
type BadParamError <: Exception
end

Base.showerror(io::IO, e::BadParamError) = print(io, "HMC sampler terminates because of too many re-runs resulted from DomainError (over $(RerunThreshold)). This may be due to large value of ϵ and τ. Please try tuning these parameters.");
