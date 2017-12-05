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
  m ~ Normal(0,sqrt.(s))
  x[1] ~ Normal(m, sqrt.(s))
  x[2] ~ Normal(m, sqrt.(s))
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
Sampler(alg::HMC) = Sampler(alg::HMC, Stan.Adapt())
Sampler(alg::HMC, adapt_conf::Stan.Adapt) = begin
  spl = Sampler(HMCDA(alg.n_iters, 0, 0.0, alg.epsilon * alg.tau, alg.space, alg.gid), adapt_conf)
  spl.info[:pre_set_ϵ] = alg.epsilon
  spl
end

Sampler(alg::Hamiltonian) =  Sampler(alg::Hamiltonian, Stan.Adapt())
Sampler(alg::Hamiltonian, adapt_conf::Stan.Adapt) = begin
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
  info[:grad_cache] = Dict{UInt64,Vector}()

  # Adapt configuration
  if adapt_conf != nothing
    info[:adapt_conf] = adapt_conf
  end

  Sampler(alg, info)
end

function prepare{T<:Hamiltonian}(model::Function, spl::Sampler{T}, vi::VarInfo)
  dprintln(3, "X-> R...")
  link!(vi, spl)
  runmodel(model, vi, spl)
  vi
end

function unprepare{T<:Hamiltonian}(model::Function, spl::Sampler{T}, vi::VarInfo)
  dprintln(3, "R -> X...")
  invlink!(vi, spl)
  cleandual!(vi)
  vi
end

function sample{T<:Hamiltonian}(model::Function, alg::T;
                                chunk_size=CHUNKSIZE,     # set temporary chunk size
                                save_state=false,         # flag for state saving
                                resume_from=nothing,      # chain to continue
                                reuse_spl_n=0,            # flag for spl re-using
                                adapt_conf=Stan.Adapt()   # adapt configuration
                               )

  default_chunk_size = CHUNKSIZE  # record global chunk size
  setchunksize(chunk_size)        # set temp chunk size

  spl = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        Sampler(alg, adapt_conf)

  @assert isa(spl.alg, Hamiltonian) "[Turing] alg type mismatch; please use resume() to re-use spl"

  alg_str = isa(alg, HMC)   ? "HMC"   :
            isa(alg, HMCDA) ? "HMCDA" :
            isa(alg, SGHMC) ? "SGHMC" :
            isa(alg, SGLD)  ? "SGLD"  :
            isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

  # Initialization
  time_total = zero(Float64)
  n = reuse_spl_n > 0 ?
      reuse_spl_n :
      alg.n_iters
  samples = Array{Sample}(n)
  weight = 1 / n
  for i = 1:n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

  vi = resume_from == nothing ?
       Base.invokelatest(model, VarInfo(), nothing) :
       deepcopy(resume_from.info[:vi])

  vi = prepare(model, spl, vi)

  # HMC steps
  if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0) end
  for i = 1:n
    dprintln(2, "$alg_str stepping...")

    time_elapsed = @elapsed vi = step(model, spl, vi, i == 1)
    time_total += time_elapsed

    if spl.info[:accept_his][end]     # accepted => store the new predcits
      samples[i].value = Sample(vi, spl).value
    else                              # rejected => store the previous predcits
      samples[i] = samples[i - 1]
    end
    samples[i].value[:elapsed] = time_elapsed
    samples[i].value[:lf_eps] = spl.info[:wum][:ϵ][end]

    if PROGRESS ProgressMeter.next!(spl.info[:progress]) end
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
  stds_str = length(stds_str) >= 32 ? stds_str[1:30]*"..." : stds_str   # only show part of pre-cond
  println("  pre-cond. diag mat  = $(stds_str).")

  setchunksize(default_chunk_size)      # revert global chunk size

  if resume_from != nothing   # concat samples
    unshift!(samples, resume_from.value2...)
  end
  c = Chain(0, samples)       # wrap the result by Chain
  if save_state               # save state
    # Convert vi back to X if vi is required to be saved
    if spl.alg.gid == 0 invlink!(vi, spl) end
    save!(c, spl, model, vi)
  end

  c
end

assume{T<:Hamiltonian}(spl::Sampler{T}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
  dprintln(2, "assuming...")
  updategid!(vi, vn, spl)
  r = vi[vn]
  acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
  r
end

assume{A<:Hamiltonian,D<:Distribution}(spl::Sampler{A}, dists::Vector{D}, vn::VarName, var::Any, vi::VarInfo) = begin
  @assert length(dists) == 1 "[observe] Turing only support vectorizing i.i.d distribution"
  dist = dists[1]
  n = size(var)[end]

  vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

  rs = vi[vns]  # NOTE: inside Turing the Julia conversion should be sticked to

  acclogp!(vi, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1]))))

  if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
    @assert size(var) == size(rs) "[assume] variable and random number dimension unmatched"
    var = rs
  elseif isa(dist, MultivariateDistribution)
    if isa(var, Vector)
      @assert length(var) == size(rs)[2] "[assume] variable and random number dimension unmatched"
      for i = 1:n
        var[i] = rs[:,i]
      end
    elseif isa(var, Matrix)
      @assert size(var) == size(rs) "[assume] variable and random number dimension unmatched"
      var = rs
    else
      error("[Turing] unsupported variable container")
    end
  end

  var
end

observe{A<:Hamiltonian}(spl::Sampler{A}, d::Distribution, value::Any, vi::VarInfo) =
  observe(nothing, d, value, vi)

observe{A<:Hamiltonian,D<:Distribution}(spl::Sampler{A}, ds::Vector{D}, value::Any, vi::VarInfo) =
  observe(nothing, ds, value, vi)
