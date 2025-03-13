"""
    ESS

Elliptical slice sampling algorithm.

# Examples
```jldoctest; setup = :(Random.seed!(1))
julia> @model function gdemo(x)
           m ~ Normal()
           x ~ Normal(m, 0.5)
       end
gdemo (generic function with 2 methods)

julia> sample(gdemo(1.0), ESS(), 1_000) |> mean
Mean

│ Row │ parameters │ mean     │
│     │ Symbol     │ Float64  │
├─────┼────────────┼──────────┤
│ 1   │ m          │ 0.824853 │
```
"""
struct ESS <: InferenceAlgorithm end

# always accept in the first step
function DynamicPPL.initialstep(
    rng::AbstractRNG, model::Model, spl::Sampler{<:ESS}, vi::AbstractVarInfo; kwargs...
)
    for vn in keys(vi)
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end
    return Transition(model, vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, spl::Sampler{<:ESS}, vi::AbstractVarInfo; kwargs...
)
    # obtain previous sample
    f = vi[:]

    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    oldstate = EllipticalSliceSampling.ESSState(f, getlogp(vi), nothing)

    # compute next state
    sample, state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(
            ESSPrior(model, spl, vi),
            Turing.LogDensityFunction(
                model, vi, DynamicPPL.SamplingContext(spl, DynamicPPL.DefaultContext())
            ),
        ),
        EllipticalSliceSampling.ESS(),
        oldstate,
    )

    # update sample and log-likelihood
    vi = DynamicPPL.unflatten(vi, sample)
    vi = setlogp!!(vi, state.loglikelihood)

    return Transition(model, vi), vi
end

# Prior distribution of considered random variable
struct ESSPrior{M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo,T}
    model::M
    sampler::S
    varinfo::V
    μ::T

    function ESSPrior{M,S,V}(
        model::M, sampler::S, varinfo::V
    ) where {M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo}
        vns = keys(varinfo)
        μ = mapreduce(vcat, vns) do vn
            dist = getdist(varinfo, vn)
            EllipticalSliceSampling.isgaussian(typeof(dist)) ||
                error("[ESS] only supports Gaussian prior distributions")
            DynamicPPL.tovec(mean(dist))
        end
        return new{M,S,V,typeof(μ)}(model, sampler, varinfo, μ)
    end
end

function ESSPrior(model::Model, sampler::Sampler{<:ESS}, varinfo::AbstractVarInfo)
    return ESSPrior{typeof(model),typeof(sampler),typeof(varinfo)}(model, sampler, varinfo)
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    sampler = p.sampler
    varinfo = p.varinfo
    # TODO: Surely there's a better way of doing this now that we have `SamplingContext`?
    vns = keys(varinfo)
    for vn in vns
        set_flag!(varinfo, vn, "del")
    end
    p.model(rng, varinfo, sampler)
    return varinfo[:]
end

# Mean of prior distribution
Distributions.mean(p::ESSPrior) = p.μ

# Evaluate log-likelihood of proposals
const ESSLogLikelihood{M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo} =
    Turing.LogDensityFunction{M,V,<:DynamicPPL.SamplingContext{<:S},AD} where {AD}

(ℓ::ESSLogLikelihood)(f::AbstractVector) = LogDensityProblems.logdensity(ℓ, f)

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, ::DefaultContext, ::Sampler{<:ESS}, right, vn, vi
)
    return DynamicPPL.tilde_assume(
        rng, LikelihoodContext(), SampleFromPrior(), right, vn, vi
    )
end

function DynamicPPL.tilde_observe(ctx::DefaultContext, ::Sampler{<:ESS}, right, left, vi)
    return DynamicPPL.tilde_observe(ctx, SampleFromPrior(), right, left, vi)
end
