struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end

function gradient_logp(
    backend::ZygoteAD,
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
        model(new_vi, sampler)
        return getlogp(new_vi)
    end

    # Compute forward and reverse passes.
    l::T, ȳ = Zygote.pullback(f, θ)
    ∂l∂θ::typeof(θ) = ȳ(1)[1]

    return l, ∂l∂θ
end

Zygote.@nograd DynamicPPL.updategid!
