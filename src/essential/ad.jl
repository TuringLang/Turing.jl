getchunksize(::AutoForwardDiff{chunk}) where {chunk} = chunk

standardtag(::AutoForwardDiff{<:Any,Nothing}) = true
standardtag(::AutoForwardDiff) = false

"""
    getADbackend(alg)

Find the autodifferentiation backend of the algorithm `alg`.
"""
getADbackend(spl::Sampler) = getADbackend(spl.alg)
getADbackend(::SampleFromPrior) = AutoForwardDiff(; chunksize=0) # TODO: remove `getADbackend`
getADbackend(ctx::DynamicPPL.SamplingContext) = getADbackend(ctx.sampler)
getADbackend(ctx::DynamicPPL.AbstractContext) = getADbackend(DynamicPPL.NodeTrait(ctx), ctx)

getADbackend(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext) = AutoForwardDiff(; chunksize=0)
getADbackend(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext) = getADbackend(DynamicPPL.childcontext(ctx))

function LogDensityProblemsAD.ADgradient(ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(getADbackend(ℓ.context), ℓ)
end

function LogDensityProblemsAD.ADgradient(ad::AutoForwardDiff, ℓ::Turing.LogDensityFunction)
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

function LogDensityProblemsAD.ADgradient(::AutoEnzyme, ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(Val(:Enzyme), ℓ)
end

function LogDensityProblemsAD.ADgradient(ad::AutoReverseDiff, ℓ::Turing.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile=Val(ad.compile), x=DynamicPPL.getparams(ℓ))
end
