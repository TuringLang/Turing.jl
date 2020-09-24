using .ReverseDiff: compile, GradientTape
using .ReverseDiff.DiffResults: GradientResult

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
    T = typeof(getlogp(vi))

    # Specify objective function.
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        model(new_vi, sampler, context)
        return getlogp(new_vi)
    end
    tp, result = taperesult(f, θ)
    ReverseDiff.gradient!(result, tp, θ)
    l = DiffResults.value(result)
    ∂l∂θ::typeof(θ) = DiffResults.gradient(result)

    return l, ∂l∂θ
end

tape(f, x) = GradientTape(f, x)
function taperesult(f, x)
    return tape(f, x), GradientResult(x)
end

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
        T = typeof(getlogp(vi))

        # Specify objective function.
        function f(θ)
            new_vi = VarInfo(vi, sampler, θ)
            model(new_vi, sampler, context)
            return getlogp(new_vi)
        end
        ctp, result = memoized_taperesult(f, θ)
        ReverseDiff.gradient!(result, ctp, θ)
        l = DiffResults.value(result)
        ∂l∂θ = DiffResults.gradient(result)

        return l, ∂l∂θ
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
        return compiledtape(k.f, k.x), GradientResult(k.x)
    end
    compiledtape(f, x) = compile(GradientTape(f, x))
end
