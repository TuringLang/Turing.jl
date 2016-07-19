immutable HMC <: InferenceAlgorithm
  n_samples ::  Int64     # number of samples
  lf_size   ::  Float64   # leapfrog step size
  lf_num    ::  Int64     # leapfrog step number
end

type HMCSampler{HMC} <: Sampler{HMC}
  alg         :: HMC
  model       :: Function
  samples     :: Array{Dict{Symbol, Any}}
  logjoint    :: Dual{Float64}
  predicts    :: Dict{Symbol, Any}
  priors      :: Dict{Symbol,Any}
  first       :: Bool

  function HMCSampler(alg :: HMC, model :: Function)
    samples = Array{Dict{Symbol, Any}}(alg.n_samples)
    for i = 1:alg.n_samples
      samples[i] = Dict{Symbol, Any}()
    end
    logjoint = Dual(0, 0)
    predicts = Dict{Symbol, Any}()
    priors = Dict{Symbol, Any}()
    new(alg, model, samples, logjoint, predicts, priors, true)
  end
end

function Base.run(spl :: Sampler{HMC})
  function get_gradient_dict(priors)
    val∇E = Dict{Symbol, Any}()
    for k in keys(priors)
      real = isa(spl.priors[k], Dual) ? realpart(spl.priors[k]) : realpart(spl.priors[k][1])
      spl.priors[k] = Dual(real, 1)
      consume(Task(spl.model))
      val∇E[k] = dualpart(-spl.logjoint)
      spl.logjoint = Dual(0, 0)
      spl.priors[k] = Dual(real, 0)
    end
    return val∇E
  end

  function half_momentum_step(p, val∇E)
    for k in keys(p)
      p[k] -= ϵ * val∇E[k] / 2
    end
    return p
  end

  # Run the model for the first time
  # println("initialising...")
  consume(Task(spl.model))
  spl.logjoint = Dual(0, 0)
  spl.first = false
  spl.predicts = Dict{Symbol,Any}()
  n = spl.alg.n_samples
  ϵ = spl.alg.lf_size
  τ = spl.alg.lf_num
  # Sampling
  for i = 1:n
    # HMC step
    # println("stepping...")
    # Generate random momentum
    p = Dict{Symbol, Any}()
    for k in keys(spl.priors)
      p[k] = randn(length(spl.priors[k]))
    end
    # Record old Hamiltonian
    oldH = 0
    for k in keys(p)
      oldH += p[k]' * p[k]
    end
    oldH += spl.logjoint
    # Record old state
    old_priors = spl.priors
    # Record old gradient
    val∇E = get_gradient_dict(spl.priors)
    # 'leapfrog' for each prior
    for t in 1:τ
      p = half_momentum_step(p, val∇E)
      # make a full step for state
      for k in keys(spl.priors)
        spl.priors[k] += ϵ * p[k]
      end
      val∇E = get_gradient_dict(spl.priors)
      p = half_momentum_step(p, val∇E)
    end
    # claculate the new Hamiltonian
    H = 0
    for k in keys(p)
      H += p[k]' * p[k]
    end
    H += spl.logjoint
    # calculate the difference in Hamiltonian
    ΔH = H - oldH
    # Vector{Any, 1} -> Any
    ΔH = ΔH[1]
    # decide wether to accept or not
    if ΔH < 0
      acc = true
    elseif rand() < exp(-ΔH)
      acc = true
    else
      acc = false
    end
    # rewind of rejected
    if ~acc
      spl.priors = old_priors
    end
    # update predicts if acc
    if acc
      for k in keys(spl.predicts)
        spl.predicts[k] = spl.priors[k]
      end
    end
    # Store the samples
    spl.samples[i] = spl.predicts
    spl.predicts = Dict{Symbol,Any}()
  end
  results = Dict{Symbol, Any}()
  results[:samples] = spl.samples
  return results
end

function assume(spl :: HMCSampler{HMC}, d :: Distribution, name :: Symbol)
  # println("assuming...")
  # println(spl.first)
  if spl.first                          # The first time running the program
    prior = Dual(rand(d), 0)            # Generate a new prior
    spl.priors[name] = prior            # Store the generated prior
  else                                  # ; not first time
    prior = spl.priors[name]            # Fetch the existing prior
  end
  # Turn Array{Any} to Any if necessary (this is due to randn())
  prior = isa(prior, Array) ? prior[1] : prior
  spl.logjoint += log(hmcpdf(d, prior))
  return prior
end

function observe(spl :: HMCSampler{HMC}, score :: Dual)
  # println("observing...")
  spl.logjoint += score
end

function predict(spl :: HMCSampler{HMC}, name :: Symbol, value)
  # println("predicting...")
  spl.predicts[name] = value
end

sample(model :: Function, alg :: HMC) = (
                                        global sampler = HMCSampler{HMC}(alg, model);
                                        run(sampler)
                                        )
