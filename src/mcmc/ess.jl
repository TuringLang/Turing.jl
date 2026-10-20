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
    M<:AbstractVector{<:Real},
}
    ldf::L
    params::P
    loglikelihood::R
    prior_means::M
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
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())

    priors = DynamicPPL.get_priors(accs)
    vector_params = DynamicPPL.get_vector_params(accs)
    loglike = DynamicPPL.getloglikelihood(accs)

    # Check that priors are all Gaussian
    for dist in values(priors)
        EllipticalSliceSampling.isgaussian(typeof(dist)) ||
            error("ESS only supports Gaussian prior distributions")
    end

    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(accs)
    means = prior_means(rng, model, loglike_ldf)
    state = TuringESSState(loglike_ldf, vector_params, loglike, means)
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
            ESSPrior(state.ldf, state.prior_means), ESSLikelihood(state.ldf)
        ),
        EllipticalSliceSampling.ESS(),
        wrapped_state,
    )

    transition = discard_sample ? nothing : DynamicPPL.ParamsWithStats(sample, state.ldf)
    new_state = TuringESSState(
        state.ldf, sample, new_wrapped_state.loglikelihood, state.prior_means
    )
    return transition, new_state
end

"""
    InitFromPriorMean()

Set each variable to the mean of its prior, evaluated with the variables it depends on already
set to theirs.
"""
struct InitFromPriorMean <: DynamicPPL.AbstractInitStrategy end
function DynamicPPL.init(
    ::Random.AbstractRNG, ::VarName, dist::Distribution, ::InitFromPriorMean
)
    return DynamicPPL.TransformedValue(mean(dist), DynamicPPL.NoTransform())
end

"""
    prior_means(rng, model, ldf)

The mean of the joint prior over the variables `ldf` samples, laid out as `ldf`'s vector.

ESS needs the mean of the joint prior, not of each variable's prior given the current values
of the others: `m ~ Normal(); x ~ Normal(m, 1)` has a joint prior with mean zero for `x`
whatever `m` is. The joint prior is Gaussian only if each variable's mean is linear in the
variables it depends on and its covariance does not depend on them, and then setting every
variable to its mean in evaluation order gives the joint mean exactly.
"""
function prior_means(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, ldf::DynamicPPL.LogDensityFunction
)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorParamAccumulator(ldf))
    _, accs = DynamicPPL.init!!(
        rng, model, accs, InitFromPriorMean(), DynamicPPL.UnlinkAll()
    )
    return DynamicPPL.get_vector_params(accs)
end

# Prior distribution of considered random variable
struct ESSPrior{L<:DynamicPPL.LogDensityFunction,T<:AbstractVector{<:Real}}
    ldf::L
    means::T
end

# Ensure that the prior is a Gaussian distribution (checked in the first step)
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

function gibbs_get_parameter_values(state::TuringESSState)
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
    # Another component can change the parameters of this block's conditional priors.
    new_ldf, new_params, accs = gibbs_recompute_ldf_and_params(
        state.ldf, model, global_vals, (DynamicPPL.LogLikelihoodAccumulator(),)
    )
    new_loglike = DynamicPPL.getloglikelihood(accs)
    # `model` is conditioned on every variable outside this block, so only the block's own
    # variables are set to their means. Setting them draws nothing, so any RNG will do.
    new_means = prior_means(Random.default_rng(), model, new_ldf)
    return TuringESSState(new_ldf, new_params, new_loglike, new_means)
end
