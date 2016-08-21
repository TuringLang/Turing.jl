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
  logjoint    :: Dual
  predicts    :: Dict{Symbol, Any}
  priors      :: Dict{Any, Any}
  first       :: Bool

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Sample}(alg.n_samples)
    weight = 1 / alg.n_samples
    for i = 1:alg.n_samples
      samples[i] = Sample(weight, Dict{Symbol, Any}())
    end
    logjoint = Dual(0)
    predicts = Dict{Symbol, Any}()
    priors = Dict{Any, Any}()
    new(alg, model, samples, logjoint, predicts, priors, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  # Function to generate the gradient dictionary
  function get_gradient_dict()
    val∇E = Dict{Any, Any}()
    prior_dim = 0
    # Get total dimension of priors
    for k in keys(spl.priors)
      prior_dim += length(spl.priors[k])
    end
    # Set dual part correspondingly
    prior_count = 1
    for k in keys(spl.priors)
      if length(spl.priors[k]) == 1
        real = isa(spl.priors[k], Dual) ? realpart(spl.priors[k]) : realpart(spl.priors[k][1])
        spl.priors[k] = make_dual(prior_dim, real, prior_count)
        prior_count += 1
      else
        l = length(spl.priors[k])
        reals = realpart(spl.priors[k])
        spl.priors[k] = []
        for i = 1:l
          push!(spl.priors[k], make_dual(prior_dim, reals[i], prior_count))
          prior_count += 1
        end
      end
    end
    # Run the model
    spl.logjoint = Dual{prior_dim, Real}(0)
    consume(Task(spl.model))
    # Collect gradient
    prior_count = 1
    for k in keys(spl.priors)
      if length(spl.priors[k]) == 1
        val∇E[k] = dualpart(-spl.logjoint)[prior_count]
        prior_count += 1
      else
        l = length(spl.priors[k])
        # To store the gradient vector
        g = zeros(l)
        for i = 1:l
          # Collect
          g[i] = dualpart(-spl.logjoint)[prior_count]
          prior_count += 1
        end
        val∇E[k] = deepcopy(g)
      end
    end
    # Reset logjoint
    spl.logjoint = Dual(0)

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
        # Get gradient dict
        val∇E = get_gradient_dict()
        # 'leapfrog' for each prior
        for t in 1:τ
          p = half_momentum_step(p, val∇E)
          # Make a full step for state
          for k in keys(spl.priors)
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

# TODO: Use another way to achieve replay. The current method fails when fetching arrays
function assume(spl :: HMCSampler{HMC}, dd :: dDistribution, p :: Prior)
  dprintln(2, "assuming...")
  name = string(p)
  # TODO: Change the first running condition
  # If it's the first time running the program
  if spl.first
    # Generate a new prior
    r = rand(dd)
    if length(r) == 1
      prior = Dual(r)
    else
      prior = map(x -> Dual(x), r)
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
    spl.logjoint += log(pdf(dd, map(x -> Dual(x), value)))
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
