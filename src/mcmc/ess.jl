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

struct TuringESSState{
    L<:DynamicPPL.LogDensityFunction,
    P<:AbstractVector{<:Real},
    R<:Real,
    Va<:DynamicPPL.VarNamedTuple,
    Vb<:DynamicPPL.VarNamedTuple,
}
    ldf::L
    params::P
    loglikelihood::R
    priors::Va
end

# always accept in the first step
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    ::ESS;
    discard_sample=false,
    initial_params,
    kwargs...,
)
    # Set up a LogDensityFunction which evaluates the model's log-likelihood.
    # Note that this costs one model evaluation (fine since it's only in the first step)
    loglike_ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getloglikelihood, DynamicPPL.UnlinkAll()
    )

    # Run the model using the specified initialisation strategy and extract all necessary
    # information.
    accs = DynamicPPL.OnlyAccsVarInfo(
        # no transforms so no need for LogJacobianAccumulator
        DynamicPPL.LogPriorAccumulator(),
        DynamicPPL.LogLikelihoodAccumulator(),
        DynamicPPL.VectorParamAccumulator(loglike_ldf),
        DynamicPPL.PriorDistributionAccumulator(),
        DynamicPPL.RawValueAccumulator(true),  # for ParamsWithStats later
    )
    _, accs = DynamicPPL.init!!(model, accs, initial_params, DynamicPPL.UnlinkAll())

    priors = DynamicPPL.get_priors(accs)
    vector_params = DynamicPPL.get_vector_params(accs)
    loglike = DynamicPPL.getloglikelihood(accs)

    # Check that priors are all Gaussian
    for dist in values(priors)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end

    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(accs)
    state = TuringESSState(loglike_ldf, vector_params, loglike, priors)
    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    ::ESS,
    state::TuringESSState;
    discard_sample=false,
    kwargs...,
)
    # define previous sampler state
    # (do not use cache to avoid in-place sampling from prior)
    wrapped_state = EllipticalSliceSampling.ESSState(
        state.params, state.loglikelihood, nothing
    )

    # compute next state
    sample, new_wrapped_state = AbstractMCMC.step(
        rng,
        EllipticalSliceSampling.ESSModel(
            ESSPrior(state.ldf, state.priors), ESSLikelihood(state.ldf)
        ),
        EllipticalSliceSampling.ESS(),
        wrapped_state,
    )

    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(sample, state.ldf)
    new_state = TuringESSState(
        state.ldf, sample, new_wrapped_state.loglikelihood, state.priors
    )
    return transition, new_state
end

# NOTE: This is a quick and easy definition but it assumes that _vec(x) is the same as
# Bijectors.VectorBijectors.from_vec(dist) for all distributions we care about in the
# priors. If that ever becomes untrue, then this could silently cause bugs.
_vec(x::Real) = [x]
_vec(x::AbstractArray) = vec(x)

# Prior distribution of considered random variable
struct ESSPrior{L<:DynamicPPL.LogDensityFunction,T<:AbstractVector{<:Real}}
    ldf::L
    means::T

    function ESSPrior(ldf::DynamicPPL.LogDensityFunction, priors::DynamicPPL.VarNamedTuple)
        # Calculate means from priors.
        means = fill(NaN, LogDensityProblems.dimension(ldf))
        for (vn, dist) in pairs(priors)
            range = DynamicPPL.get_range_and_transform(ldf, vn).range
            this_mean = _vec(mean(dist))
            means[range] .= this_mean
        end
        if any(isnan, means)
            error(
                "Some means were not filled in when constructing ESSPrior. This is likely a bug in Turing.jl, please report it.",
            )
        end
        return new{typeof(ldf),typeof(means)}(ldf, means)
    end
end

# Ensure that the prior is a Gaussian distribution (checked in the constructor)
EllipticalSliceSampling.isgaussian(::Type{<:ESSPrior}) = true

# Only define out-of-place sampling
function Base.rand(rng::Random.AbstractRNG, p::ESSPrior)
    return Base.rand(rng, p.ldf)
end

# Mean of prior distribution
Distributions.mean(p::ESSPrior) = p.means

# Evaluate log-likelihood of proposals. We need this struct because
# EllipticalSliceSampling.jl expects a callable struct / a function as its
# likelihood.
struct ESSLikelihood{L<:DynamicPPL.LogDensityFunction}
    ldf::L

    # Force usage of `getloglikelihood` in inner constructor
    function ESSLikelihood(ldf::DynamicPPL.LogDensityFunction)
        logp_callable = DynamicPPL.get_logdensity_callable(ldf)
        if logp_callable !== DynamicPPL.getloglikelihood
            error(
                "The log-density function passed to ESSLikelihood must use `getloglikelihood` as its log-density function, but found $(logp_callable). This is likely a bug in Turing.jl, please report it!",
            )
        end
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

####
#### Gibbs interface
####

function gibbs_get_raw_values(state::TuringESSState)
    pws = DynamicPPL.ParamsWithStats(
        state.params, state.ldf; include_log_probs=false, include_colon_eq=false
    )
    return pws.params
end

function gibbs_update_state!!(
    ::ESS,
    state::TuringESSState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    # We need to update everything in `state` except for the priors (which are constant). We
    # pass an extra LogLikelihoodAccumulator here so that we can calculate the new loglike in
    # one pass.
    new_ldf, new_params, accs = gibbs_recompute_ldf_and_params(
        state.ldf, model, global_vals, (DynamicPPL.LogLikelihoodAccumulator(),)
    )
    new_loglike = DynamicPPL.getloglikelihood(accs)
    return TuringESSState(new_ldf, new_params, new_loglike, state.priors)
end
