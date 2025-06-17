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

DynamicPPL.initialsampler(::ESS) = DynamicPPL.SampleFromPrior()
update_sample_kwargs(::ESS, ::Integer, kwargs) = kwargs
get_adtype(::ESS) = nothing
requires_unconstrained_space(::ESS) = false

# always accept in the first step
function AbstractMCMC.step(rng::AbstractRNG, ldf::LogDensityFunction, spl::ESS; kwargs...)
    vi = ldf.varinfo
    for vn in keys(vi)
        dist = getdist(vi, vn)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end
    return Transition(ldf.model, vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, ldf::LogDensityFunction, spl::ESS, vi::AbstractVarInfo; kwargs...
)
    # obtain previous sample
    f = vi[:]

    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    oldstate = EllipticalSliceSampling.ESSState(f, getlogp(vi), nothing)

    # compute next state
    # Note: `f_loglikelihood` effectively calculates the log-likelihood (not
    # log-joint, despite the use of `LDP.logdensity`) because `tilde_assume` is
    # overloaded on `SamplingContext(rng, ESS(), ...)` below.
    f_loglikelihood = Base.Fix1(LogDensityProblems.logdensity, ldf)
    sample, state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(ESSPrior(ldf.model, spl, vi), f_loglikelihood),
        EllipticalSliceSampling.ESS(),
        oldstate,
    )

    # update sample and log-likelihood
    vi = DynamicPPL.unflatten(vi, sample)
    vi = setlogp!!(vi, state.loglikelihood)

    return Transition(ldf.model, vi), vi
end

# Prior distribution of considered random variable
struct ESSPrior{M<:Model,V<:AbstractVarInfo,T}
    model::M
    sampler::ESS
    varinfo::V
    μ::T

    function ESSPrior{M,V}(
        model::M, sampler::ESS, varinfo::V
    ) where {M<:Model,V<:AbstractVarInfo}
        vns = keys(varinfo)
        μ = mapreduce(vcat, vns) do vn
            dist = getdist(varinfo, vn)
            EllipticalSliceSampling.isgaussian(typeof(dist)) ||
                error("[ESS] only supports Gaussian prior distributions")
            DynamicPPL.tovec(mean(dist))
        end
        return new{M,V,typeof(μ)}(model, sampler, varinfo, μ)
    end
end

function ESSPrior(model::Model, sampler::ESS, varinfo::AbstractVarInfo)
    return ESSPrior{typeof(model),typeof(varinfo)}(model, sampler, varinfo)
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    # TODO(penelopeysm): This is ugly -- need to set 'del' flag because
    # otherwise DynamicPPL.SampleWithPrior will just use the existing
    # parameters in the varinfo. In general SampleWithPrior etc. need to be
    # reworked.
    for vn in keys(p.varinfo)
        set_flag!(p.varinfo, vn, "del")
    end
    _, vi = DynamicPPL.evaluate!!(p.model, p.varinfo, SamplingContext(rng, p.sampler))
    return vi[:]
end

# Mean of prior distribution
Distributions.mean(p::ESSPrior) = p.μ

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, ::DefaultContext, ::ESS, right, vn, vi
)
    return DynamicPPL.tilde_assume(
        rng, LikelihoodContext(), SampleFromPrior(), right, vn, vi
    )
end
