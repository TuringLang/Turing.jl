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
    rng::AbstractRNG, model::Model, ::Sampler{<:ESS}, vi::AbstractVarInfo; kwargs...
)
    for vn in keys(vi)
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end
    return Transition(model, vi, nothing), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, ::Sampler{<:ESS}, vi::AbstractVarInfo; kwargs...
)
    # obtain previous sample
    f = vi[:]

    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    oldstate = EllipticalSliceSampling.ESSState(f, DynamicPPL.getloglikelihood(vi), nothing)

    # compute next state
    sample, state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(ESSPrior(model, vi), ESSLikelihood(model, vi)),
        EllipticalSliceSampling.ESS(),
        oldstate,
    )

    # update sample and log-likelihood
    vi = DynamicPPL.unflatten(vi, sample)
    vi = DynamicPPL.setloglikelihood!!(vi, state.loglikelihood)

    return Transition(model, vi, nothing), vi
end

# Prior distribution of considered random variable
struct ESSPrior{M<:Model,V<:AbstractVarInfo,T}
    model::M
    varinfo::V
    μ::T

    function ESSPrior(model::Model, varinfo::AbstractVarInfo)
        vns = keys(varinfo)
        μ = mapreduce(vcat, vns) do vn
            dist = getdist(varinfo, vn)
            EllipticalSliceSampling.isgaussian(typeof(dist)) ||
                error("[ESS] only supports Gaussian prior distributions")
            DynamicPPL.tovec(mean(dist))
        end
        return new{typeof(model),typeof(varinfo),typeof(μ)}(model, varinfo, μ)
    end
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    _, vi = DynamicPPL.init!!(rng, p.model, p.varinfo, DynamicPPL.InitFromPrior())
    return vi[:]
end

# Mean of prior distribution
Distributions.mean(p::ESSPrior) = p.μ

# Evaluate log-likelihood of proposals. We need this struct because
# EllipticalSliceSampling.jl expects a callable struct / a function as its
# likelihood.
struct ESSLikelihood{L<:DynamicPPL.LogDensityFunction}
    ldf::L

    # Force usage of `getloglikelihood` in inner constructor
    function ESSLikelihood(model::Model, varinfo::AbstractVarInfo)
        ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getloglikelihood, varinfo)
        return new{typeof(ldf)}(ldf)
    end
end

(ℓ::ESSLikelihood)(f::AbstractVector) = LogDensityProblems.logdensity(ℓ.ldf, f)
