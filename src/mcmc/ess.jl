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

struct TuringESSState{V<:DynamicPPL.AbstractVarInfo,VNT<:DynamicPPL.VarNamedTuple}
    vi::V
    priors::VNT
end
get_varinfo(state::TuringESSState) = state.vi

# always accept in the first step
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    ::ESS;
    discard_sample=false,
    initial_params,
    kwargs...,
)
    vi = DynamicPPL.VarInfo()
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.RawValueAccumulator(true))
    prior_acc = DynamicPPL.PriorDistributionAccumulator()
    prior_accname = DynamicPPL.accumulator_name(prior_acc)
    vi = DynamicPPL.setacc!!(vi, prior_acc)
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params, DynamicPPL.UnlinkAll())
    priors = DynamicPPL.getacc(vi, Val(prior_accname)).values

    for dist in values(priors)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end
    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, model)
    return transition, TuringESSState(vi, priors)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    ::ESS,
    state::TuringESSState;
    discard_sample=false,
    kwargs...,
)
    # obtain previous sample
    vi = state.vi
    f = vi[:]

    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    wrapped_state = EllipticalSliceSampling.ESSState(
        f, DynamicPPL.getloglikelihood(vi), nothing
    )

    # compute next state
    sample, new_wrapped_state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(
            ESSPrior(model, vi, state.priors), ESSLikelihood(model, vi)
        ),
        EllipticalSliceSampling.ESS(),
        wrapped_state,
    )

    # update sample and log-likelihood
    vi = DynamicPPL.unflatten!!(vi, sample)
    vi = DynamicPPL.setloglikelihood!!(vi, new_wrapped_state.loglikelihood)

    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, model)
    return transition, TuringESSState(vi, state.priors)
end

# Prior distribution of considered random variable
struct ESSPrior{M<:Model,V<:AbstractVarInfo,T}
    model::M
    varinfo::V
    μ::T

    function ESSPrior(
        model::Model, varinfo::AbstractVarInfo, priors::DynamicPPL.VarNamedTuple
    )
        μ = mapreduce(vcat, priors; init=Float64[]) do pair
            prior_dist = pair.second
            EllipticalSliceSampling.isgaussian(typeof(prior_dist)) || error(
                "[ESS] only supports Gaussian prior distributions, but found $(typeof(prior_dist))",
            )
            DynamicPPL.tovec(mean(prior_dist))
        end
        return new{typeof(model),typeof(varinfo),typeof(μ)}(model, varinfo, μ)
    end
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    _, vi = DynamicPPL.init!!(
        rng, p.model, p.varinfo, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
    )
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
