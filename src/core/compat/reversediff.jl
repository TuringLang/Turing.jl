using .ReverseDiff: compile, GradientTape
using .ReverseDiff.DiffResults: GradientResult

struct ReverseDiffAD{cache} <: ADBackend end
const RDCache = Ref(false)
setcache(b::Bool) = RDCache[] = b
getcache() = RDCache[]
ADBackend(::Val{:reversediff}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:reverse_diff})
    Base.depwarn("`Turing.setadbackend(:reverse_diff)` is deprecated. Please use `Turing.setadbackend(:tracker)` to use `Tracker` or `Turing.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.",  :setadbackend)
    setadbackend(Val(:reversediff))
end
function setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
end

function gradient_logp(
    backend::ReverseDiffAD{cache},
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
) where {cache}
    T = typeof(getlogp(vi))
    
    # Specify objective function.
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        return getlogp(runmodel!(model, new_vi, sampler))
    end
    if cache
        ctp, result = memoized_taperesult(f, θ)
        ReverseDiff.gradient!(result, ctp, θ)
    else
        tp, result = taperesult(f, θ)
        ReverseDiff.gradient!(result, tp, θ)
    end

    l = DiffResults.value(result)
    ∂l∂θ = DiffResults.gradient(result)

    return l, ∂l∂θ
end

# This makes sure we generate a single tape per Turing model and sampler
struct RDTapeKey{F, Tx}
	f::F
	x::Tx
end
function Memoization._get!(f, d, keys::Tuple{Tuple{RDTapeKey}, Nothing})
    key = keys[1][1]
	return Memoization._get!(f, d, (typeof(key.f), typeof(key.x), size(key.x)))
end

tape(f, x) = GradientTape(f, x)
compiledtape(f, x) = compile(GradientTape(f, x))
function taperesult(f, x)
    return tape(f, x), GradientResult(x)
end

memoized_taperesult(f, x) = memoized_taperesult(RDTapeKey(f, x))
@memoize function memoized_taperesult(k::RDTapeKey)
    return compiledtape(k.f, k.x), GradientResult(k.x)
end
memoized_tape(f, x) = memoized_tape(RDTapeKey(f, x))
@memoize memoized_tape(k::RDTapeKey) = compiledtape(k.f, k.x)
