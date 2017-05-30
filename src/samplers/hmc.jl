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
immutable HMC <: Hamiltonian
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

# NOTE: the implementation of HMC is removed,
#       it now reuses the one of HMCDA
Sampler(alg::HMC) = begin
  spl = Sampler(HMCDA(alg.n_iters, 0, 0.0, alg.epsilon * alg.tau, alg.space, alg.gid))
  spl.info[:pre_set_ϵ] = alg.epsilon
  spl
end

Sampler(alg::Hamiltonian) = begin
  info=Dict{Symbol, Any}()

  # For sampler infomation
  info[:accept_his] = []
  info[:lf_num] = 0
  info[:total_lf_num] = 0
  info[:total_eval_num] = 0

  # For pre-conditioning
  info[:θ_mean] = nothing
  info[:θ_num] = 0
  info[:stds] = nothing
  info[:vars] = nothing

  # For caching gradient
  info[:grad_cache] = Dict{Vector,Vector}()
  Sampler(alg, info)
end

sample(model::Function, alg::Hamiltonian) = sample(model, alg, CHUNKSIZE)

# NOTE: in the previous code, `sample` would call `run`; this is
# now simplified: `sample` and `run` are merged into one function.
function sample{T<:Hamiltonian}(model::Function, alg::T, chunk_size::Int)
  default_chunk_size = CHUNKSIZE
  global CHUNKSIZE = chunk_size

  spl = Sampler(alg);
  alg_str = isa(alg, HMC)   ? "HMC"   :
            isa(alg, HMCDA) ? "HMCDA" :
            isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

  # Initialization
  time_total = zero(Float64)
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

    time_elapsed = @elapsed vi = step(model, spl, vi, i==1)
    time_total += time_elapsed

    if spl.info[:accept_his][end]     # accepted => store the new predcits
      samples[i].value = Sample(vi, spl).value
    else                              # rejected => store the previous predcits
      samples[i] = samples[i - 1]
    end
    samples[i].value[:elapsed] = time_elapsed
    ProgressMeter.next!(spl.info[:progress])
  end

  println("[$alg_str] Finished with")
  println("  Running time        = $time_total;")
  if ~isa(alg, NUTS)  # accept rate for NUTS is meaningless - so no printing
    accept_rate = sum(spl.info[:accept_his]) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")
  end
  println("  #lf / sample        = $(spl.info[:total_lf_num] / n);")
  println("  #evals / sample     = $(spl.info[:total_eval_num] / n);")
  stds_str = string(spl.info[:wum][:stds])
  stds_str = length(stds_str) >= 16 ? stds_str[1:14]*"..." : stds_str
  println("  pre-cond. diag mat  = $stds_str.")

  global CHUNKSIZE = default_chunk_size

  Chain(0, samples)    # wrap the result by Chain
end

assume{T<:Hamiltonian}(spl::Sampler{T}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
  dprintln(2, "assuming...")
  updategid!(vi, vn, spl)
  r = vi[vn]
  acclogp!(vi, logpdf(dist, r, istrans(vi, vn)))
  r
end

assume{A<:Hamiltonian,D<:Distribution}(spl::Sampler{A}, dists::Vector{D}, vn::VarName, variable::Any, vi::VarInfo) = begin
  @assert length(dists) == 1 "[observe] Turing only support vectorizing i.i.d distribution"
  dist = dists[1]
  n = size(variable)[end]

  vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

  rs = vi[vns]

  acclogp!(vi, sum(logpdf(dist, rs, istrans(vi, vns[1]))))

  rs
end

observe{A<:Hamiltonian}(spl::Sampler{A}, d::Distribution, value::Any, vi::VarInfo) =
  observe(nothing, d, value, vi)

observe{A<:Hamiltonian,D<:Distribution}(spl::Sampler{A}, ds::Vector{D}, value::Any, vi::VarInfo) =
  observe(nothing, ds, value, vi)
