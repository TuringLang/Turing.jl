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
    return Transition(model, vi), vi
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

    return Transition(model, vi), vi
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
    varinfo = p.varinfo
    # TODO: Surely there's a better way of doing this now that we have `SamplingContext`?
    # TODO(DPPL0.37/penelopeysm): This can be replaced with `init!!(p.model,
    # p.varinfo, PriorInit())` after TuringLang/DynamicPPL.jl#984. The reason
    # why we had to use the 'del' flag before this was because
    # SampleFromPrior() wouldn't overwrite existing variables.
    # The main problem I'm rather unsure about is ESS-within-Gibbs. The
    # current implementation I think makes sure to only resample the variables
    # that 'belong' to the current ESS sampler. InitContext on the other hand
    # would resample all variables in the model (??) Need to think about this
    # carefully.
    vns = keys(varinfo)
    for vn in vns
        set_flag!(varinfo, vn, "del")
    end
    p.model(rng, varinfo)
    return varinfo[:]
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
