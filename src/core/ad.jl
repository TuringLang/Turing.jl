"""
gradient(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Function,
    sampler::Union{Nothing, Sampler}=nothing,
)

Computes the gradient of the log joint of `θ` for the model specified by
`(vi, sampler, model)` using whichever automatic differentation tool is currently active.
"""
function gradient(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Function,
    sampler::Union{Nothing, Sampler}=nothing;
    backend::Symbol=ADBACKEND[],
    is_negativelogp=true
)

    # Record old parameters.
    vals_old, logp_old = copy(vi.vals), copy(vi.logp)

    # Specify objective function.
    f(θ) = is_negativelogp ?
        begin
            vi[sampler] = θ
            return -runmodel!(model, vi, sampler).logp
        end :
        begin
            vi[sampler] = θ
            return runmodel!(model, vi, sampler).logp
        end

    @assert backend ∈ (:forward_diff, :reverse_diff)
    if backend == :forward_diff
        ∂l∂θ = gradient_forward(θ, f)
        l = vi.logp.value
    else
        l, ∂l∂θ = gradient_reverse(θ, f)
    end

    # Replace old parameters to ensure this function doesn't mutate `vi`.
    vi.vals, vi.logp = vals_old, logp_old

    return l, ∂l∂θ 
end

"""
gradient_forward(
    θ::AbstractVector{<:Real},
    f::Function
    chunk_size::Int=CHUNKSIZE[],
)

Computes the gradient of the log joint of `θ` for the model specified by `(vi, spl, model)`
using forwards-mode AD from ForwardDiff.jl.
"""
function gradient_forward(
    θ::AbstractVector{<:Real},
    f::Function,
    chunk_size::Int=CHUNKSIZE[],
)
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ∂l∂θ = ForwardDiff.gradient!(similar(θ), f, θ, config)

    # Strip tracking info from θ to avoid mutating it.
    θ .= ForwardDiff.value.(θ)

    return ∂l∂θ
end

"""
gradient_reverse(
    θ::AbstractVector{<:Real},
    f::Function
)

Computes the gradient of the log joint of `θ` for the model specified by
`(vi, sampler, model)` using reverse-mode AD from Flux.jl.
"""
function gradient_reverse(
    θ::AbstractVector{<:Real},
    f::Function
)

    # Compute forward and reverse passes.
    l_tracked, ȳ = Tracker.forward(f, θ)
    l, ∂l∂θ = Tracker.data(l_tracked), Tracker.data(ȳ(1)[1])

    # Strip tracking info from θ to avoid mutating it.
    θ .= Tracker.data.(θ)

    # Return non-tracked gradient value
    return l, ∂l∂θ
end

import Base: <=
<=(a::Tracker.TrackedReal, b::Tracker.TrackedReal) = a.data <= b.data

function verifygrad(grad::AbstractVector{<:Real})
    if any(isnan.(grad)) || any(isinf.(grad))
        @warn("Numerical error has been found in gradients.")
        @warn("grad = $(grad)")
        return false
    else
        return true
    end
end
