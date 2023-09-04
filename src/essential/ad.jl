##############################
# Global variables/constants #
##############################
const ADBACKEND = Ref(:forwarddiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
function setadbackend(backend::Val)
    _setadbackend(backend)
    AdvancedVI.setadbackend(backend)
end

function _setadbackend(::Val{:forwarddiff})
    ADBACKEND[] = :forwarddiff
end
function _setadbackend(::Val{:tracker})
    @warn "Usage of Tracker.jl with Turing.jl is no longer being actively tested and maintained; please use at your own risk. See Zygote.jl or ReverseDiff.jl for fully supported reverse-mode backends."
    ADBACKEND[] = :tracker
end
function _setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end
function _setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
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

getchunksize(::ForwardDiffAD{chunk}) where chunk = chunk

standardtag(::ForwardDiffAD{<:Any,true}) = true
standardtag(::ForwardDiffAD) = false

struct TrackerAD <: ADBackend end
struct ZygoteAD <: ADBackend end

struct ReverseDiffAD{cache} <: ADBackend end

const RDCache = Ref(false)

setrdcache(b::Bool) = setrdcache(Val(b))
setrdcache(::Val{false}) = RDCache[] = false
setrdcache(::Val{true}) = RDCache[] = true

getrdcache() = RDCache[]

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))

ADBackend(::Val{:forwarddiff}) = ForwardDiffAD{CHUNKSIZE[]}
ADBackend(::Val{:tracker}) = TrackerAD
ADBackend(::Val{:zygote}) = ZygoteAD
ADBackend(::Val{:reversediff}) = ReverseDiffAD{getrdcache()}

ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")

"""
    getADbackend(alg)

Find the autodifferentiation backend of the algorithm `alg`.
"""
getADbackend(spl::Sampler) = getADbackend(spl.alg)
getADbackend(::SampleFromPrior) = ADBackend()()
getADbackend(ctx::DynamicPPL.SamplingContext) = getADbackend(ctx.sampler)
getADbackend(ctx::DynamicPPL.AbstractContext) = getADbackend(DynamicPPL.NodeTrait(ctx), ctx)

getADbackend(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext) = ADBackend()()
getADbackend(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext) = getADbackend(DynamicPPL.childcontext(ctx))

function LogDensityProblemsAD.ADgradient(ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(getADbackend(ℓ.context), ℓ)
end

function LogDensityProblemsAD.ADgradient(ad::ForwardDiffAD, ℓ::Turing.LogDensityFunction)
    θ = DynamicPPL.getparams(ℓ)
    f = Base.Fix1(LogDensityProblems.logdensity, ℓ)

    # Define configuration for ForwardDiff.
    tag = if standardtag(ad)
        ForwardDiff.Tag(Turing.TuringTag(), eltype(θ))
    else
        ForwardDiff.Tag(f, eltype(θ))
    end
    chunk_size = getchunksize(ad)
    chunk = if chunk_size == 0
        ForwardDiff.Chunk(θ)
    else
        ForwardDiff.Chunk(length(θ), chunk_size)
    end

    return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; chunk, tag, x = θ)
end

function LogDensityProblemsAD.ADgradient(::TrackerAD, ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(Val(:Tracker), ℓ)
end

function LogDensityProblemsAD.ADgradient(::ZygoteAD, ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(Val(:Zygote), ℓ)
end

for cache in (:true, :false)
    @eval begin
        function LogDensityProblemsAD.ADgradient(::ReverseDiffAD{$cache}, ℓ::Turing.LogDensityFunction)
            return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile=Val($cache))
        end
    end
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
