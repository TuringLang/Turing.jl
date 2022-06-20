using .ReverseDiff: compile, GradientTape

struct ReverseDiffAD{cache} <: ADBackend end
const RDCache = Ref(false)
setrdcache(b::Bool) = setrdcache(Val(b))
setrdcache(::Val{false}) = RDCache[] = false
setrdcache(::Val) = throw("Memoization.jl is not loaded. Please load it before setting the cache to true.")
function emptyrdcache end

getrdcache() = RDCache[]
ADBackend(::Val{:reversediff}) = ReverseDiffAD{getrdcache()}
function _setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
end

function gradient_logp(
    backend::ReverseDiffAD{false},
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    # Save current log density value.
    logp_old = getlogp(vi)

    # Define log density function.
    f = Turing.LogDensityFunction(vi, model, sampler, context)

    # Obtain both value and gradient of the log density function.
    tp, result = taperesult(f, θ)
    ReverseDiff.gradient!(result, tp, θ)
    logp = DiffResults.value(result)
    ∂logp∂θ = DiffResults.gradient(result)

    # Ensure that `vi` was not mutated.
    @assert getlogp(vi) == logp_old

    return logp, ∂logp∂θ
end

tape(f, x) = GradientTape(f, x)
taperesult(f, x) = (tape(f, x), DiffResults.GradientResult(x))

@require Memoization = "6fafb56a-5788-4b4e-91ca-c0cea6611c73" @eval begin
    setrdcache(::Val{true}) = RDCache[] = true
    function emptyrdcache()
        Memoization.empty_cache!(memoized_taperesult)
        return
    end

    function gradient_logp(
        backend::ReverseDiffAD{true},
        θ::AbstractVector{<:Real},
        vi::VarInfo,
        model::Model,
        sampler::AbstractSampler = SampleFromPrior(),
        context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
    )
        # Save current log density value.
        logp_old = getlogp(vi)

        # Define log density function.
        f = Turing.LogDensityFunction(vi, model, sampler, context)

        # Obtain both value and gradient of the log density function.
        ctp, result = memoized_taperesult(f, θ)
        ReverseDiff.gradient!(result, ctp, θ)
        logp = DiffResults.value(result)
        ∂logp∂θ = DiffResults.gradient(result)

        # Ensure that `vi` was not mutated.
        @assert getlogp(vi) == logp_old

        return logp, ∂logp∂θ
    end

    # This makes sure we generate a single tape per Turing model and sampler
    struct RDTapeKey{F, Tx}
        f::F
        x::Tx
    end
    function Memoization._get!(f, d::Dict, keys::Tuple{Tuple{RDTapeKey}, Any})
        key = keys[1][1]
        return Memoization._get!(f, d, (key.f, typeof(key.x), size(key.x), Threads.threadid()))
    end
    memoized_taperesult(f, x) = memoized_taperesult(RDTapeKey(f, x))
    Memoization.@memoize Dict function memoized_taperesult(k::RDTapeKey)
        return compiledtape(k.f, k.x), DiffResults.GradientResult(k.x)
    end
    compiledtape(f, x) = compile(GradientTape(f, x))
end
