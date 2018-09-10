"""
gradient_forward(
  θ::AbstractVector{<:Real},
  vi::VarInfo,
  model::Function,
  spl::Union{Nothing, Sampler}=nothing,
  chunk_size::Int=CHUNKSIZE[],
)

Computes the gradient of the log joint of `θ` for the model specified by `(vi, spl, model)`
using forwards-mode AD from ForwardDiff.jl.
"""
function gradient_forward(
  θ::AbstractVector{<:Real},
  vi::VarInfo,
  model::Function,
  spl::Union{Nothing, Sampler}=nothing,
  chunk_size::Int=CHUNKSIZE[],
)
  # Record old parameters.
  θ_old = vi[spl]

  # Define function to compute log joint.
  function f(θ)
      vi[spl] = θ
      runmodel(model, vi, spl)
      return -vi.logp
  end

  # Set chunk size and do ForwardMode.
  chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
  config = ForwardDiff.GradientConfig(f, θ, chunk)
  ∂l∂θ = ForwardDiff.gradient!(similar(θ), f, θ, config)

  # Replace old parameters to ensure this function doesn't mutate `vi`.
  vi[spl] = θ_old

  return getlogp(vi), ∂l∂θ
end

"""
gradient_reverse(
  θ::AbstractVector{<:Real},
  vi::VarInfo,
  model::Function,
  spl::Union{Nothing, Sampler}=nothing,
)

Computes the gradient of the log joint of `θ` for the model specified by `(vi, spl, model)`
using reverse-mode AD from Flux.jl.
"""
function gradient_reverse(
  θ::AbstractVector{<:Real},
  vi::Turing.VarInfo,
  model::Function,
  spl::Union{Nothing, Sampler}=nothing,
)
  # Specify objective function.
  function f(θ)
    vi[spl] = θ
    runmodel(model, vi, spl)
    return -vi.logp
  end

  # Compute forward and reverse passes.
  l, ȳ = Tracker.forward(f, θ)
  ∂l∂θ = ȳ(1)[1]

  # Remove tracking info from variables in model (because mutable state).
  vi.logp = Tracker.data(vi.logp)
  vi[spl] .= Tracker.data.(vi[spl])

  # Return non-tracked gradient value
  return Tracker.data(l), Tracker.data(∂l∂θ)
end

function verifygrad(grad::AbstractVector{<:Real})
  if any(isnan.(grad)) || any(isinf.(grad))
    @warn("Numerical error has been found in gradients.")
    @warn("grad = $(grad)")
    return false
  else
    return true
  end
end
