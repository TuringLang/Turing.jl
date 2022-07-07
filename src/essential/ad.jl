##############################
# Global variables/constants #
##############################
const ADBACKEND = Ref(:forwarddiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
function setadbackend(backend::Val)
    _setadbackend(backend)
    AdvancedVI.setadbackend(backend)
    Bijectors.setadbackend(backend)
end

function _setadbackend(::Val{:forwarddiff})
    ADBACKEND[] = :forwarddiff
end
function _setadbackend(::Val{:tracker})
    ADBACKEND[] = :tracker
end
function _setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end

const ADSAFE = Ref(false)
function setadsafe(switch::Bool)
    @info("[Turing]: global ADSAFE is set as $switch")
    ADSAFE[] = switch
end

const CHUNKSIZE = Ref(0) # 0 means letting ForwardDiff set it automatically

function setchunksize(chunk_size::Int)
    @info("[Turing]: AD chunk size is set as $chunk_size")
    CHUNKSIZE[] = chunk_size
    AdvancedVI.setchunksize(chunk_size)
end

abstract type ADBackend end
struct ForwardDiffAD{chunk,standardtag} <: ADBackend end

# Use standard tag if not specified otherwise
ForwardDiffAD{N}() where {N} = ForwardDiffAD{N,true}()

getchunksize(::Type{<:ForwardDiffAD{chunk}}) where chunk = chunk
getchunksize(::Type{<:Sampler{Talg}}) where Talg = getchunksize(Talg)
getchunksize(::Type{SampleFromPrior}) = CHUNKSIZE[]

standardtag(::ForwardDiffAD{<:Any,true}) = true
standardtag(::ForwardDiffAD) = false

struct TrackerAD <: ADBackend end
struct ZygoteAD <: ADBackend end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))

ADBackend(::Val{:forwarddiff}) = ForwardDiffAD{CHUNKSIZE[]}
ADBackend(::Val{:tracker}) = TrackerAD
ADBackend(::Val{:zygote}) = ZygoteAD
ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")

"""
    getADbackend(alg)

Find the autodifferentiation backend of the algorithm `alg`.
"""
getADbackend(spl::Sampler) = getADbackend(spl.alg)
getADbackend(spl::SampleFromPrior) = ADBackend()()

"""
    gradient_logp(
        θ::AbstractVector{<:Real},
        vi::AbstractVarInfo,
        model::Model,
        sampler::AbstractSampler,
        ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
    )

Computes the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using whichever automatic differentation
tool is currently active.
"""
function gradient_logp(
    θ::AbstractVector{<:Real},
    vi::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler,
    ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    return gradient_logp(getADbackend(sampler), θ, vi, model, sampler, ctx)
end

"""
gradient_logp(
    backend::ADBackend,
    θ::AbstractVector{<:Real},
    vi::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
    ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)

Compute the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using `backend` for AD, e.g. `ForwardDiffAD{N}()` uses `ForwardDiff.jl` with chunk size `N`, `TrackerAD()` uses `Tracker.jl` and `ZygoteAD()` uses `Zygote.jl`.
"""
function gradient_logp(
    ad::ForwardDiffAD,
    θ::AbstractVector{<:Real},
    vi::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    # Define log density function.
    f = Turing.LogDensityFunction(vi, model, sampler, context)

    # Define configuration for ForwardDiff.
    tag = if standardtag(ad)
        ForwardDiff.Tag(Turing.TuringTag(), eltype(θ))
    else
        ForwardDiff.Tag(f, eltype(θ))
    end
    chunk_size = getchunksize(typeof(ad))
    config = if chunk_size == 0
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(θ), tag)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size), tag)
    end

    # Obtain both value and gradient of the log density function.
    out = DiffResults.GradientResult(θ)
    ForwardDiff.gradient!(out, f, θ, config)
    logp = DiffResults.value(out)
    ∂logp∂θ = DiffResults.gradient(out)

    return logp, ∂logp∂θ
end
function gradient_logp(
    ::TrackerAD,
    θ::AbstractVector{<:Real},
    vi::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    # Define log density function.
    f = Turing.LogDensityFunction(vi, model, sampler, context)

    # Compute forward pass and pullback.
    l_tracked, ȳ = Tracker.forward(f, θ)

    # Remove tracking info.
    l::typeof(getlogp(vi)) = Tracker.data(l_tracked)
    ∂l∂θ::typeof(θ) = Tracker.data(only(ȳ(1)))

    return l, ∂l∂θ
end

function gradient_logp(
    backend::ZygoteAD,
    θ::AbstractVector{<:Real},
    vi::AbstractVarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    # Define log density function.
    f = Turing.LogDensityFunction(vi, model, sampler, context)

    # Compute forward pass and pullback.
    l::typeof(getlogp(vi)), ȳ = ZygoteRules.pullback(f, θ)
    ∂l∂θ::typeof(θ) = only(ȳ(1))

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

# These still seem necessary
for F in (:link, :invlink)
    @eval begin
        $F(dist::PDMatDistribution, x::Tracker.TrackedArray) = Tracker.track($F, dist, x)
        Tracker.@grad function $F(dist::PDMatDistribution, x::Tracker.TrackedArray)
            x_data = Tracker.data(x)
            T = eltype(x_data)
            y = $F(dist, x_data)
            return  y, Δ -> begin
                out = reshape((ForwardDiff.jacobian(x -> $F(dist, x), x_data)::Matrix{T})' * vec(Δ), size(Δ))
                return (nothing, out)
            end
        end
    end
end
