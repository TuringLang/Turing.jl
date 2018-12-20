abstract type ADBackend end
struct ForwardDiffAD{chunk} <: ADBackend end
struct FluxTrackerAD <: ADBackend end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))
function ADBackend(::Val{T}) where {T}
    if T === :forward_diff
        return ForwardDiffAD{CHUNKSIZE[]}
    else
        return FluxTrackerAD
    end
end

getchunksize(::Type{ForwardDiffAD{chunk}}) where chunk = Val(chunk)


"""
getADtype(alg)

Finds the autodifferentiation type of the algorithm `alg`.
"""
getADtype() = ADBackend()
getADtype(s::Sampler) = getADtype(typeof(s))
getADtype(s::Type{<:Sampler{TAlg}}) where {TAlg} = getADtype(TAlg)
getADtype(alg::Hamiltonian) = getADtype(typeof(alg))
getADtype(::Type{<:Hamiltonian{AD}}) where {AD} = AD

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
@generated function gradient(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Function,
    sampler::TS=nothing,
) where {TS <: Union{Nothing, Sampler}}
    if TS == Nothing
        return quote
            ad_type = getADtype()
            if ad_type <: ForwardDiffAD
                gradient_forward(θ, vi, model, sampler, getchunksize(ad_type))
            else
                gradient_reverse(θ, vi, model, sampler)
            end
        end
    else
        ad_type = getADtype(TS)
        @assert any(T -> ad_type <: T, (ForwardDiffAD, FluxTrackerAD))
        if ad_type <: ForwardDiffAD 
            chunk = getchunksize(ad_type)
            return :(gradient_forward(θ, vi, model, sampler, $chunk))
        else ad_type <: FluxTrackerAD
            return :(gradient_reverse(θ, vi, model, sampler))
        end
    end
end

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
    sampler::Union{Nothing, Sampler}=nothing,
    ::Val{chunk_size}=Val(CHUNKSIZE[]),
) where chunk_size
    # Record old parameters.
    vals_old, logp_old = copy(vi.vals), copy(vi.logp)

    # Define function to compute log joint.
    function f(θ)
        vi[sampler] = θ
        return -runmodel!(model, vi, sampler).logp
    end

    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ∂l∂θ = ForwardDiff.gradient!(similar(θ), f, θ, config)
    l = vi.logp.value

    # Replace old parameters to ensure this function doesn't mutate `vi`.
    vi.vals, vi.logp = vals_old, logp_old

    # Strip tracking info from θ to avoid mutating it.
    θ .= ForwardDiff.value.(θ)

    return l, ∂l∂θ
end

"""
gradient_reverse(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Function,
    sampler::Union{Nothing, Sampler}=nothing,
)

Computes the gradient of the log joint of `θ` for the model specified by
`(vi, sampler, model)` using reverse-mode AD from Flux.jl.
"""
function gradient_reverse(
    θ::AbstractVector{<:Real},
    vi::Turing.VarInfo,
    model::Function,
    sampler::Union{Nothing, Sampler}=nothing,
)
    vals_old, logp_old = copy(vi.vals), copy(vi.logp)

    # Specify objective function.
    function f(θ)
        vi[sampler] = θ
        return -runmodel!(model, vi, sampler).logp
    end

    # Compute forward and reverse passes.
    l_tracked, ȳ = Tracker.forward(f, θ)
    l, ∂l∂θ = Tracker.data(l_tracked), Tracker.data(ȳ(1)[1])

    # Remove tracking info from variables in model (because mutable state).
    vi.vals, vi.logp = vals_old, logp_old

    # Strip tracking info from θ to avoid mutating it.
    θ .= Tracker.data.(θ)

    # Return non-tracked gradient value
    return l, ∂l∂θ
end

import Base: <=
<=(a::Tracker.TrackedReal, b::Tracker.TrackedReal) = a.data <= b.data

function verifygrad(grad::AbstractVector{<:Real})
    if any(isnan, grad) || any(isinf, grad)
        @warn("Numerical error has been found in gradients.")
        @warn("grad = $(grad)")
        return false
    else
        return true
    end
end

import StatsFuns: binomlogpdf
binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int) = Tracker.track(binomlogpdf, n, p, x)
Tracker.@grad function binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int)
    return binomlogpdf(n, Tracker.data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end


import StatsFuns: poislogpdf
poislogpdf(v::Tracker.TrackedReal, x::Int) = Tracker.track(poislogpdf, v, x)
Tracker.@grad function poislogpdf(v::Tracker.TrackedReal, x::Int)
      return poislogpdf(Tracker.data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
end
