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
struct ESS <: AbstractSampler end

# always accept in the first step
function Turing.Inference.initialstep(
    rng::AbstractRNG, model::DynamicPPL.Model, ::ESS, vi::AbstractVarInfo; kwargs...
)
    for vn in keys(vi)
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end
    return Transition(model, vi, nothing), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, ::ESS, vi::AbstractVarInfo; kwargs...
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

# Needed for method ambiguity resolution, even though this method is never going to be
# called in practice. This just shuts Aqua up.
# TODO(penelopeysm): Remove this when the default `step(rng, ::DynamicPPL.Model,
# ::AbstractSampler) method in `src/mcmc/abstractmcmc.jl` is removed.
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::EllipticalSliceSampling.ESS;
    kwargs...,
)
    return error(
        "This method is not implemented! If you want to use the ESS sampler in Turing.jl, please use `Turing.ESS()` instead. If you want the default behaviour in EllipticalSliceSampling.jl, wrap your model in a different subtype of `AbstractMCMC.AbstractModel`, and then implement the necessary EllipticalSliceSampling.jl methods on it.",
    )
end
