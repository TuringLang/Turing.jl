doc"""
    HMC(n_iters::Int, epsilon::Float64, tau::Int)

Hamiltonian Monte Carlo sampler.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))
```
"""
immutable HMC <: InferenceAlgorithm
  n_iters   ::  Int       # number of samples
  epsilon   ::  Float64   # leapfrog step size
  tau       ::  Int       # leapfrog step number
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int       # group ID
  HMC(epsilon::Float64, tau::Int, space...) = HMC(1, epsilon, tau, space..., 0)
  HMC(n_iters::Int, epsilon::Float64, tau::Int) = new(n_iters, epsilon, tau, Set(), 0)
  HMC(n_iters::Int, epsilon::Float64, tau::Int, space...) =
    new(n_iters, epsilon, tau, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  HMC(alg::HMC, new_gid::Int) = new(alg.n_iters, alg.epsilon, alg.tau, alg.space, new_gid)
end

typealias Hamiltonian Union{HMC,HMCDA,NUTS}

# NOTE: the implementation of HMC is removed,
#       it now reuses the one of HMCDA
Sampler(alg::HMC) = begin
  spl = Sampler(HMCDA(alg.n_iters, 0, 0.0, alg.epsilon * alg.tau, alg.space, alg.gid))
  spl.info[:ϵ] = [alg.epsilon]
  spl
end

Sampler(alg::Hamiltonian) = begin
  info=Dict{Symbol, Any}()
  info[:accept_his] = []
  info[:lf_num] = 0
  info[:eval_num] = 0
  Sampler(alg, info)
end

sample(model::Function, alg::Hamiltonian) = sample(model, alg, CHUNKSIZE)

# NOTE: in the previous code, `sample` would call `run`; this is
# now simplified: `sample` and `run` are merged into one function.
function sample{T<:Hamiltonian}(model::Function, alg::T, chunk_size::Int)
  global CHUNKSIZE = chunk_size;
  spl = Sampler(alg);
  alg_str = isa(alg, HMC)   ? "HMC"   :
            isa(alg, HMCDA) ? "HMCDA" :
            isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

  # Initialization
  n =  spl.alg.n_iters
  samples = Array{Sample}(n)
  weight = 1 / n
  for i = 1:n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end
  vi = model()

  if spl.alg.gid == 0
    link!(vi, spl)
    runmodel(model, vi, spl)
  end

  # HMC steps
  spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0)
  for i = 1:n
    dprintln(2, "$alg_str stepping...")
    vi = step(model, spl, vi, i==1)
    if spl.info[:accept_his][end]     # accepted => store the new predcits
      samples[i].value = Sample(vi).value
    else                              # rejected => store the previous predcits
      samples[i] = samples[i - 1]
    end
    ProgressMeter.next!(spl.info[:progress])
  end

  if ~isa(alg, NUTS)  # cccept rate for NUTS is meaningless - so no printing
    accept_rate = sum(spl.info[:accept_his]) / n  # calculate the accept rate
    log_str = """
    [$alg_str] Done with accept rate     = $accept_rate;
                         #lf / sample    = $(spl.info[:lf_num] / n);
                         #evals / sample = $(spl.info[:eval_num] / n).
    """
    println(log_str)
  end

  Chain(0, samples)    # wrap the result by Chain
end

function assume{T<:Hamiltonian}(spl::Sampler{T}, dist::Distribution, vn::VarName, vi::VarInfo)
  dprintln(2, "assuming...")
  updategid!(vi, vn, spl)
  r = vi[vn]
  acclogp!(vi, logpdf(dist, r, istrans(vi, vn)))
  r
end

function observe{T<:Hamiltonian}(spl::Sampler{T}, d::Distribution, value::Any, vi::VarInfo)
  dprintln(2, "observing...")
  acclogp!(vi, logpdf(d, map(x -> Dual(x), value)))
end
