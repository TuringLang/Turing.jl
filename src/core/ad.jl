##############################
# Global variables/constants #
##############################

const ADBACKEND = Ref(:forward_diff)
function setadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
    backend_sym == :forward_diff && CHUNKSIZE[] == 0 && setchunksize(40)
    ADBACKEND[] = backend_sym
end

const ADSAFE = Ref(false)
function setadsafe(switch::Bool)
    @info("[Turing]: global ADSAFE is set as $switch")
    ADSAFE[] = switch
end

const CHUNKSIZE = Ref(40) # default chunksize used by AD

function setchunksize(chunk_size::Int)
    if ~(CHUNKSIZE[] == chunk_size)
        @info("[Turing]: AD chunk size is set as $chunk_size")
        CHUNKSIZE[] = chunk_size
    end
end

abstract type ADBackend end
struct ForwardDiffAD{chunk} <: ADBackend end
getchunksize(::T) where {T <: ForwardDiffAD} = getchunksize(T)
getchunksize(::Type{ForwardDiffAD{chunk}}) where chunk = chunk
getchunksize(::T) where {T <: Sampler} = getchunksize(T)
getchunksize(::Type{<:Sampler{T}}) where {T} = getchunksize(T)
getchunksize(::SampleFromPrior) = getchunksize(Nothing)
getchunksize(::Type{Nothing}) = CHUNKSIZE[]

struct TrackerAD <: ADBackend end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))
function ADBackend(::Val{T}) where {T}
    if T === :forward_diff
        return ForwardDiffAD{CHUNKSIZE[]}
    else
        return TrackerAD
    end
end

"""
getADtype(alg)

Finds the autodifferentiation type of the algorithm `alg`.
"""
getADtype(::Nothing) = getADtype(Nothing)
getADtype(::Type{Nothing}) = getADtype()
getADtype() = ADBackend()
getADtype(s::Sampler) = getADtype(typeof(s))
getADtype(s::Type{<:Sampler{TAlg}}) where {TAlg} = getADtype(TAlg)

"""
gradient_logp(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
)

Computes the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using whichever automatic differentation
tool is currently active.
"""
function gradient_logp(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::TS,
) where {TS <: Sampler}

    ad_type = getADtype(TS)
    if ad_type <: ForwardDiffAD
        return gradient_logp_forward(θ, vi, model, sampler)
    else ad_type <: TrackerAD
        return gradient_logp_reverse(θ, vi, model, sampler)
    end
end

"""
gradient_logp_forward(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    spl::AbstractSampler=SampleFromPrior(),
)

Computes the value of the log joint of `θ` and its gradient for the model
specified by `(vi, spl, model)` using forwards-mode AD from ForwardDiff.jl.
"""
function gradient_logp_forward(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
)
    # Record old parameters.
    vals_old, logp_old = copy(vi.vals), copy(vi.logp)

    # Define function to compute log joint.
    function f(θ)
        vi[sampler] = θ
        return runmodel!(model, vi, sampler).logp
    end

    chunk_size = getchunksize(sampler)
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
gradient_logp_reverse(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
)

Computes the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using reverse-mode AD from Tracker.jl.
"""
function gradient_logp_reverse(
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
)
    vals_old, logp_old = copy(vi.vals), copy(vi.logp)

    # Specify objective function.
    function f(θ)
        vi[sampler] = θ
        return runmodel!(model, vi, sampler).logp
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

function verifygrad(grad::AbstractVector{<:Real})
    if any(isnan, grad) || any(isinf, grad)
        @warn("Numerical error in gradients. Rejecting current proposal...")
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

import StatsFuns: nbinomlogpdf
# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

nbinomlogpdf(n::Tracker.TrackedReal, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Tracker.TrackedReal, p::Real, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Real, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Tracker._zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Tracker._zero(p), nothing)
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

function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)

    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = Δ_p + Δ_r
    return FD(nbinomlogpdf(val_r, val_p, k),  Δ)
end
function nbinomlogpdf(r::Real, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(r, val_p, k)
    return FD(nbinomlogpdf(r, val_p, k),  Δ_p)
end
function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::Real, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, p, k)
    return FD(nbinomlogpdf(val_r, p, k),  Δ_r)
end

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
end

fleshout(M::AbstractMatrix) = [M[i,j] for i in 1:size(M,1), j in 1:size(M,2)]
fleshout(v::AbstractVector) = [v[i] for i in 1:length(v)]

function Distributions.MvNormal(u::AbstractVector, M::Union{Tracker.TrackedVector, Tracker.TrackedMatrix})
    return MvNormal(u, fleshout(M))
end
for T in (:(Tracker.TrackedVector{T}), :(Tracker.TrackedMatrix{T}))
    @eval begin
        function Distributions.MvNormal(u::Tracker.TrackedVector{T}, M::$T) where {T <: Real}
            return MvNormal(fleshout(u), fleshout(M))
        end
    end
end
for T in (:(Vector{T}), :(Matrix{T}), :(AbstractVector{T}), :(AbstractMatrix{T}))
    @eval begin
        function Distributions.MvNormal(u::Tracker.TrackedVector{T}, M::$T) where {T <: Real}
            return MvNormal(fleshout(u), M)
        end
    end
end

LinearAlgebra.cholesky(M::Tracker.TrackedMatrix) = cholesky(fleshout(M))
