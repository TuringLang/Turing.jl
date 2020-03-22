struct ReverseDiffAD <: ADBackend end
ADBackend(::Val{:reversediff}) = ReverseDiffAD
function setadbackend(::Val{:reverse_diff})
    @warn("Turing.setadbackend(:reverse_diff) is deprecated. Please use `Turing.setadbackend(:tracker)` to use `Tracker` or `Turing.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.")
    setadbackend(Val(:reversediff))
end
function setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
end

function gradient_logp(
    backend::ReverseDiffAD,
    θ::AbstractVector{<:Real},
    vi::VarInfo,
    model::Model,
    sampler::AbstractSampler = SampleFromPrior(),
)
    T = typeof(getlogp(vi))
    
    # Specify objective function.
    function f(θ)
        new_vi = VarInfo(vi, sampler, θ)
        return getlogp(runmodel!(model, new_vi, sampler))
    end

    ∂l∂θ = similar(θ)
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(∂l∂θ, tp, θ)
    l = ReverseDiff.value(tp.output)

    return l, ∂l∂θ
end
