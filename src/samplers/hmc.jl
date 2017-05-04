doc"""
    HMC(n_iters::Int, epsilon::Float64, tau::Int)

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
  n_iters   ::  Int       # number of samples
  epsilon   ::  Float64   # leapfrog step size
  tau       ::  Int       # leapfrog step number
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int
  function HMC(epsilon::Float64, tau::Int, space...)
    HMC(1, epsilon, tau, space..., 0)
  end
  function HMC(n_iters, epsilon, tau)
    new(n_iters, epsilon, tau, Set(), 0)
  end
  function HMC(n_iters, epsilon, tau, space...)
    space = isa(space, Symbol) ? Set([space]) : Set(space)
    new(n_iters, epsilon, tau, space, 0)
  end
  HMC(alg::HMC, new_gid::Int) = new(alg.n_iters, alg.epsilon, alg.tau, alg.space, new_gid)
end

typealias Hamiltonian Union{HMC,HMCDA,NUTS}

Sampler(alg::HMC) = begin
  info = Dict{Symbol, Any}()
  info[:ϵ] = [alg.epsilon]
  Sampler(HMCDA(alg.n_iters, 0, 0.0, alg.epsilon * alg.tau, alg.space, alg.gid), info)
end

Sampler(alg::Hamiltonian) = begin
  info = Dict{Symbol, Any}()
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

  # initialization
  n =  spl.alg.n_iters
  samples = Array{Sample}(n)
  weight = 1 / n
  for i = 1:n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end
  accept_num = 0    # record the accept number
  vi = model()

  if spl.alg.gid == 0 vi = link(vi, spl) end

  # HMC steps
  spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0)
  for i = 1:n
    dprintln(2, "recording old θ...")
    old_vi = deepcopy(vi)
    dprintln(2, "$alg_str stepping...")
    is_accept, vi = step(model, spl, vi, i==1)
    if is_accept    # accepted => store the new predcits
      samples[i].value = Sample(vi).value
      accept_num = accept_num + 1
    else            # rejected => store the previous predcits
      vi = old_vi
      samples[i] = samples[i - 1]
    end
    ProgressMeter.next!(spl.info[:progress])
  end

  accept_rate = accept_num / n    # calculate the accept rate

  println("[$alg_str] Done with accept rate = $accept_rate.")

  Chain(0, samples)    # wrap the result by Chain
end

function assume{T<:Hamiltonian}(spl::Sampler{T}, dist::Distribution, vn::VarName, vi::VarInfo)
  dprintln(2, "assuming...")
  updategid!(vi, vn, spl)
  r = vi[vn]
  vi.logp += logpdf(dist, r, istransformed(vi, vn))
  r
end

function observe{T<:Hamiltonian}(spl::Sampler{T}, d::Distribution, value, vi::VarInfo)
  dprintln(2, "observing...")
  vi.logp += logpdf(d, map(x -> Dual(x), value))
end
