# Concrete algorithm implementations.
include("support/helper.jl")
include("support/resample.jl")
@suppress_err begin
  include("support/transform.jl")
end
include("support/hmc_core.jl")
include("hmcda.jl")
include("nuts.jl")
include("hmc.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("gibbs.jl")

## Fallback functions

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

assume(spl::Sampler, dist::Distribution) =
  error("[assume]: unmanaged inference algorithm: $(typeof(spl))")

observe(spl::Sampler, weight::Float64) =
  error("[observe]: unmanaged inference algorithm: $(typeof(spl))")

## Default definitions for assume, observe, when sampler = nothing.
assume(spl::Void, dist::Distribution, vn::VarName, vi::VarInfo) = begin
  if haskey(vi, vn)
    r = vi[vn]
  else
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
  end
  acclogp!(vi, logpdf(dist, r, istrans(vi, vn)))
  r
end

observe(spl::Void, dist::Distribution, value::Any, vi::VarInfo) = begin
  lp = logpdf(dist, value)
  vi.logw += lp
  acclogp!(vi, lp)
end
