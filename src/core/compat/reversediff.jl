struct ReverseDiffAD <: ADBackend end
ADBackend(::Val{:reverse_diff}) = ReverseDiffAD
function setadbackend(::Val{:reverse_diff})
    ADBACKEND[] = :reverse_diff
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
