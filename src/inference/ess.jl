"""
    ESS

Elliptical slice sampling algorithm.

# Examples
```jldoctest; setup = :(Random.seed!(1))
julia> @model gdemo(x) = begin
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
struct ESS{space} <: InferenceAlgorithm end

ESS() = ESS{()}()
ESS(space::Symbol) = ESS{(space,)}()

# always accept in the first step
function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ESS},
    vi::AbstractVarInfo;
    kwargs...
)
    # Sanity check
    vns = _getvns(vi, spl)
    length(vns) == 1 ||
        error("[ESS] does only support one variable ($(length(vns)) variables specified)")
    for vn in vns[1]
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("[ESS] only supports Gaussian prior distributions")
    end

    return Transition(vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:ESS},
    vi::AbstractVarInfo;
    kwargs...
)
    # obtain previous sample
    f = vi[spl]

    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    oldstate = EllipticalSliceSampling.ESSState(f, getlogp(vi), nothing)

    # compute next state
    sample, state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(
            ESSPrior(model, spl, vi), ESSLogLikelihood(model, spl, vi),
        ),
        EllipticalSliceSampling.ESS(),
        oldstate,
    )

    # update sample and log-likelihood
    vi[spl] = sample
    setlogp!(vi, state.loglikelihood)

    return Transition(vi), vi
end

# Prior distribution of considered random variable
struct ESSPrior{M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo,T}
    model::M
    sampler::S
    varinfo::V
    μ::T
    
    function ESSPrior{M,S,V}(model::M, sampler::S, varinfo::V) where {
        M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo
    }
        vns = _getvns(varinfo, sampler)
        μ = mapreduce(vcat, vns[1]) do vn
            dist = getdist(varinfo, vn)
            EllipticalSliceSampling.isgaussian(typeof(dist)) ||
                error("[ESS] only supports Gaussian prior distributions")
            vectorize(dist, mean(dist))
        end
        return new{M,S,V,typeof(μ)}(model, sampler, varinfo, μ)
    end
end

function ESSPrior(model::Model, sampler::Sampler{<:ESS}, varinfo::AbstractVarInfo)
    return ESSPrior{typeof(model),typeof(sampler),typeof(varinfo)}(
        model, sampler, varinfo,
    )
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    sampler = p.sampler
    varinfo = p.varinfo
    vns = _getvns(varinfo, sampler)
    for vn in Iterators.flatten(values(vns))
        set_flag!(varinfo, vn, "del")
    end
    p.model(rng, varinfo, sampler)
    return varinfo[sampler]
end

# Mean of prior distribution
Distributions.mean(p::ESSPrior) = p.μ

# Evaluate log-likelihood of proposals
struct ESSLogLikelihood{M<:Model,S<:Sampler{<:ESS},V<:AbstractVarInfo}
    model::M
    sampler::S
    varinfo::V
end

function (ℓ::ESSLogLikelihood)(f)
    sampler = ℓ.sampler
    varinfo = ℓ.varinfo
    varinfo[sampler] = f
    ℓ.model(varinfo, sampler)
    return getlogp(varinfo)
end

function DynamicPPL.tilde_assume(rng::Random.AbstractRNG, ctx::DefaultContext, sampler::Sampler{<:ESS}, right, vn, inds, vi)
    if inspace(vn, sampler)
        return DynamicPPL.tilde_assume(rng, LikelihoodContext(), SampleFromPrior(), right, vn, inds, vi)
    else
        return DynamicPPL.tilde_assume(rng, ctx, SampleFromPrior(), right, vn, inds, vi)
    end
end

function DynamicPPL.tilde_observe(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return DynamicPPL.tilde_observe(ctx, SampleFromPrior(), right, left, vi)
end

function DynamicPPL.dot_tilde_assume(rng::Random.AbstractRNG, ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vns, inds, vi)
    # TODO: Or should we do `all(Base.Fix2(inspace, sampler), vns)`?
    if inspace(first(vns), sampler)
        return DynamicPPL.dot_tilde_assume(rng, LikelihoodContext(), SampleFromPrior(), right, left, vns, inds, vi)
    else
        return DynamicPPL.dot_tilde_assume(rng, ctx, SampleFromPrior(), right, left, vns, inds, vi)
    end
end

function DynamicPPL.dot_tilde_observe(ctx::DefaultContext, sampler::Sampler{<:ESS}, right, left, vi)
    return DynamicPPL.dot_tilde_observe(ctx, SampleFromPrior(), right, left, vi)
end
