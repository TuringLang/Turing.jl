##############################
# Global variables/constants #
##############################
const ADBACKEND = Ref(:forwarddiff)
function setadbackend(backend_sym::Symbol)
    setadbackend(Val(backend_sym))
    AdvancedVI.setadbackend(Val(backend_sym))
    Bijectors.setadbackend(Val(backend_sym))
end
function setadbackend(::Val{:forward_diff})
    Base.depwarn("`Turing.setadbackend(:forward_diff)` is deprecated. Please use `Turing.setadbackend(:forwarddiff)` to use `ForwardDiff`.", :setadbackend)
    setadbackend(Val(:forwarddiff))
end
function setadbackend(::Val{:forwarddiff})
    CHUNKSIZE[] == 0 && setchunksize(40)
    ADBACKEND[] = :forwarddiff
end

function setadbackend(::Val{:reverse_diff})
    Base.depwarn("`Turing.setadbackend(:reverse_diff)` is deprecated. Please use `Turing.setadbackend(:tracker)` to use `Tracker` or `Turing.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.",  :setadbackend)
    setadbackend(Val(:tracker))
end
function setadbackend(::Val{:tracker})
    ADBACKEND[] = :tracker
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
getchunksize(::Type{<:ForwardDiffAD{chunk}}) where chunk = chunk
getchunksize(::Type{<:Sampler{Talg}}) where Talg = getchunksize(Talg)
getchunksize(::Type{SampleFromPrior}) = CHUNKSIZE[]

struct TrackerAD <: ADBackend end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))

ADBackend(::Val{:forwarddiff}) = ForwardDiffAD{CHUNKSIZE[]}
ADBackend(::Val{:tracker}) = TrackerAD
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
    sampler::AbstractSampler,
    ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    return gradient_logp(getADbackend(sampler), θ, vi, model, sampler, ctx)
end

"""
gradient_logp(
    backend::ADBackend,
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
)

Compute the value of the log joint of `θ` and its gradient for the model
specified by `(vi, sampler, model)` using `backend` for AD, e.g. `ForwardDiffAD{N}()` uses `ForwardDiff.jl` with chunk size `N`, `TrackerAD()` uses `Tracker.jl` and `ZygoteAD()` uses `Zygote.jl`.
"""
function gradient_logp(
    ::ForwardDiffAD,
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler=SampleFromPrior(),
    ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    # Define function to compute log joint.
    logp_old = getlogp(vi)
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = getlogp(new_vi)
        setlogp!(vi, ForwardDiff.value(logp))
        return logp
    end

    chunk_size = getchunksize(typeof(sampler))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ∂l∂θ = ForwardDiff.gradient!(similar(θ), f, θ, config)
    l = getlogp(vi)
    setlogp!(vi, logp_old)

    return l, ∂l∂θ
end
function gradient_logp(
    ::TrackerAD,
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
    ctx::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    T = typeof(getlogp(vi))

    # Specify objective function.
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        return getlogp(new_vi)
    end

    # Compute forward and reverse passes.
    l_tracked, ȳ = Tracker.forward(f, θ)
    # Remove tracking info from variables in model (because mutable state).
    l::T, ∂l∂θ::typeof(θ) = Tracker.data(l_tracked), Tracker.data(ȳ(1)[1])

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
