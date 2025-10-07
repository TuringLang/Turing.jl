abstract type Hamiltonian <: InferenceAlgorithm end
abstract type StaticHamiltonian <: Hamiltonian end
abstract type AdaptiveHamiltonian <: Hamiltonian end
default_num_warmup(::AdaptiveHamiltonian, N::Int) = min(1000, N ÷ 2)

###
### Sampler states
###

struct HMCState{
    TV<:AbstractVarInfo,
    TKernel<:AHMC.HMCKernel,
    THam<:AHMC.Hamiltonian,
    PhType<:AHMC.PhasePoint,
    TAdapt<:AHMC.Adaptation.AbstractAdaptor,
}
    vi::TV
    nadapts_done::Int
    kernel::TKernel
    hamiltonian::THam
    z::PhType
    adaptor::TAdapt
end

###
### Hamiltonian Monte Carlo samplers.
###

get_varinfo(state::HMCState) = state.vi

"""
    HMC(ϵ::Float64, n_leapfrog::Int; adtype::ADTypes.AbstractADType = AutoForwardDiff())

Hamiltonian Monte Carlo sampler with static trajectory.

# Arguments

- `ϵ`: The leapfrog step size to use.
- `n_leapfrog`: The number of leapfrog steps to use.
- `adtype`: The automatic differentiation (AD) backend.
    If not specified, `ForwardDiff` is used, with its `chunksize` automatically determined.

# Usage

```julia
HMC(0.05, 10)
```

# Tips

If you are receiving gradient errors when using `HMC`, try reducing the leapfrog step size `ϵ`, e.g.

```julia
# Original step size
sample(gdemo([1.5, 2]), HMC(0.1, 10), 1000)

# Reduced step size
sample(gdemo([1.5, 2]), HMC(0.01, 10), 1000)
```
"""
struct HMC{AD,metricT<:AHMC.AbstractMetric} <: StaticHamiltonian
    ϵ::Float64 # leapfrog step size
    n_leapfrog::Int # leapfrog step number
    adtype::AD
end

function HMC(
    ϵ::Float64,
    n_leapfrog::Int,
    ::Type{metricT};
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
) where {metricT<:AHMC.AbstractMetric}
    return HMC{typeof(adtype),metricT}(ϵ, n_leapfrog, adtype)
end
function HMC(
    ϵ::Float64,
    n_leapfrog::Int;
    metricT=AHMC.UnitEuclideanMetric,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    return HMC(ϵ, n_leapfrog, metricT; adtype=adtype)
end

DynamicPPL.initialsampler(::Sampler{<:Hamiltonian}) = SampleFromUniform()

function find_initial_params(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo,
    hamiltonian::AHMC.Hamiltonian;
    max_attempts::Int=1000,
)
    varinfo = deepcopy(varinfo)  # Don't mutate

    for attempts in 1:max_attempts
        theta = varinfo[:]
        z = AHMC.phasepoint(rng, theta, hamiltonian)
        isfinite(z) && return varinfo, z

        attempts == 10 &&
            @warn "failed to find valid initial parameters in $(attempts) tries; consider providing explicit initial parameters using the `initial_params` keyword"

        # Resample and try again.
        # NOTE: varinfo has to be linked to make sure this samples in unconstrained space
        varinfo = last(
            DynamicPPL.evaluate_and_sample!!(
                rng, model, varinfo, DynamicPPL.SampleFromUniform()
            ),
        )
    end

    # if we failed to find valid initial parameters, error
    return error(
        "failed to find valid initial parameters in $(max_attempts) tries. See https://turinglang.org/docs/uri/initial-parameters for common causes and solutions. If the issue persists, please open an issue at https://github.com/TuringLang/Turing.jl/issues",
    )
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Hamiltonian},
    vi_original::AbstractVarInfo;
    initial_params=nothing,
    nadapts=0,
    verbose::Bool=true,
    kwargs...,
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    vi = DynamicPPL.link(vi_original, model)

    # Extract parameters.
    theta = vi[:]

    # Create a Hamiltonian.
    metricT = getmetricT(spl.alg)
    metric = metricT(length(theta))
    ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.alg.adtype
    )
    lp_func = Base.Fix1(LogDensityProblems.logdensity, ldf)
    lp_grad_func = Base.Fix1(LogDensityProblems.logdensity_and_gradient, ldf)
    hamiltonian = AHMC.Hamiltonian(metric, lp_func, lp_grad_func)

    # If no initial parameters are provided, resample until the log probability
    # and its gradient are finite. Otherwise, just use the existing parameters.
    vi, z = if initial_params === nothing
        find_initial_params(rng, model, vi, hamiltonian)
    else
        vi, AHMC.phasepoint(rng, theta, hamiltonian)
    end
    theta = vi[:]

    # Find good eps if not provided one
    if iszero(spl.alg.ϵ)
        ϵ = AHMC.find_good_stepsize(rng, hamiltonian, theta)
        verbose && @info "Found initial step size" ϵ
    else
        ϵ = spl.alg.ϵ
    end
    # Generate a kernel and adaptor.
    kernel = make_ahmc_kernel(spl.alg, ϵ)
    adaptor = AHMCAdaptor(spl.alg, hamiltonian.metric; ϵ=ϵ)

    transition = Transition(model, vi, NamedTuple())
    state = HMCState(vi, 0, kernel, hamiltonian, z, adaptor)

    return transition, state
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:AdaptiveHamiltonian},
    state::HMCState;
    num_warmup,
    kwargs...,
)
    # Compute transition.
    hamiltonian = state.hamiltonian
    z = state.z
    t = AHMC.transition(rng, hamiltonian, state.kernel, z)

    # Adaptation
    hamiltonian, kernel, _ = AHMC.adapt!(
        hamiltonian,
        state.kernel,
        state.adaptor,
        state.nadapts_done + 1,
        num_warmup,
        t.z.θ,
        t.stat.acceptance_rate,
    )

    # Update variables
    vi = state.vi
    if t.stat.is_accept
        vi = DynamicPPL.unflatten(vi, t.z.θ)
    end

    # Compute next transition and state.
    transition = Transition(model, vi, t)
    newstate = HMCState(vi, state.nadapts_done + 1, kernel, hamiltonian, t.z, state.adaptor)

    return transition, newstate
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    state::HMCState;
    kwargs...,
)
    # Compute transition.
    hamiltonian = state.hamiltonian
    z = state.z
    t = AHMC.transition(rng, hamiltonian, state.kernel, z)

    # Update variables
    vi = state.vi
    if t.stat.is_accept
        vi = DynamicPPL.unflatten(vi, t.z.θ)
    end

    # Compute next transition and state.
    transition = Transition(model, vi, t)
    newstate = HMCState(
        vi, state.nadapts_done, state.kernel, hamiltonian, t.z, state.adaptor
    )
    return transition, newstate
end

function get_hamiltonian(model, spl, vi, state, n)
    metric = gen_metric(n, spl, state)
    ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.alg.adtype
    )
    lp_func = Base.Fix1(LogDensityProblems.logdensity, ldf)
    lp_grad_func = Base.Fix1(LogDensityProblems.logdensity_and_gradient, ldf)
    return AHMC.Hamiltonian(metric, lp_func, lp_grad_func)
end

_NADAPTS_DOCSTRING = """
!!! note
    In Turing <= v0.40, there was also had a field `n_adapts` to specify the number of
adaptation steps. This has been removed in v0.41; please use the `num_warmup` keyword
argument in the `sample` function instead.

    Likewise, the `discard_adapt` keyword argument in `sample` used to work only with adaptive Hamiltonian samplers like NUTS. This has been removed; please use the `discard_initial` keyword argument instead (which works with all samplers).

    For information on how to use these keyword arguments, please see: https://turinglang.org/docs/usage/sampling-options/#thinning-and-warmup.
"""

"""
    HMCDA(
        δ::Float64,
        λ::Float64,
        ϵ::Float64 = 0.0,
        metric::Type{<:AHMC.AbstractMetric} = AHMC.UnitEuclideanMetric;
        adtype::ADTypes.AbstractADType = Turing.DEFAULT_ADTYPE
    )

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

# Arguments


$(_NADAPTS_DOCSTRING)

# Reference

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
Research 15, no. 1 (2014): 1593-1623.
"""
struct HMCDA{AD<:ADTypes.AbstractADType,metricT<:AHMC.AbstractMetric} <: AdaptiveHamiltonian
    "target acceptance ratio"
    δ::Float64
    "target leapfrog length"
    λ::Float64
    "initial step size"
    ϵ::Float64
    "automatic differentiation backend; Turing's default is ForwardDiff"
    adtype::AD

    function HMCDA(
        δ::Float64,
        λ::Float64,
        init_ϵ::Float64=0.0,
        metric::Type{metricT}=AHMC.UnitEuclideanMetric;
        adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
    ) where {metricT<:AHMC.AbstractMetric}
        return new{typeof(adtype),metricT}(δ, λ, init_ϵ, adtype)
    end
end

"""
    NUTS(
        δ::Float64 = 0.65;
        max_depth::Int = 10,
        Δ_max::Float64 = 1000.0,
        init_ϵ::Float64 = 0.0,
        metric::Type{<:AHMC.AbstractMetric} = AHMC.DiagEuclideanMetric,
        adtype::ADTypes.AbstractADType = Turing.DEFAULT_ADTYPE
    )

No-U-Turn Sampler (NUTS) sampler.

$(TYPEDFIELDS)

$(_NADAPTS_DOCSTRING)
"""
struct NUTS{AD<:ADTypes.AbstractADType,metricT<:AHMC.AbstractMetric} <: AdaptiveHamiltonian
    "target acceptance rate"
    δ::Float64
    "maximum tree depth"
    max_depth::Int
    "maximum divergence during doubling tree"
    Δ_max::Float64
    "initial step size; use 0 to automatically search using a heuristic"
    ϵ::Float64
    "automatic differentiation backend; Turing's default is ForwardDiff"
    adtype::AD

    function NUTS(
        δ::Float64=0.65;
        max_depth::Int=10,
        Δ_max::Float64=1000.0,
        ϵ::Float64=0.0,
        metric::Type{metricT}=AHMC.DiagEuclideanMetric,
        adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
    ) where {metricT<:AHMC.AbstractMetric}
        return new{AD,metricT}(δ, max_depth, Δ_max, ϵ, adtype)
    end
end

for alg in (:HMC, :HMCDA, :NUTS)
    @eval getmetricT(::$alg{<:Any,metricT}) where {metricT} = metricT
end

#####
##### HMC core functions
#####

getstepsize(sampler::Sampler{<:Hamiltonian}, state) = sampler.alg.ϵ
getstepsize(sampler::Sampler{<:AdaptiveHamiltonian}, state) = AHMC.getϵ(state.adaptor)
function getstepsize(
    sampler::Sampler{<:AdaptiveHamiltonian},
    state::HMCState{TV,TKernel,THam,PhType,AHMC.Adaptation.NoAdaptation},
) where {TV,TKernel,THam,PhType}
    return state.kernel.τ.integrator.ϵ
end

gen_metric(dim::Int, spl::Sampler{<:Hamiltonian}, state) = AHMC.UnitEuclideanMetric(dim)
function gen_metric(dim::Int, spl::Sampler{<:AdaptiveHamiltonian}, state)
    return AHMC.renew(state.hamiltonian.metric, AHMC.getM⁻¹(state.adaptor.pc))
end

function make_ahmc_kernel(alg::HMC, ϵ)
    return AHMC.HMCKernel(
        AHMC.Trajectory{AHMC.EndPointTS}(AHMC.Leapfrog(ϵ), AHMC.FixedNSteps(alg.n_leapfrog))
    )
end
function make_ahmc_kernel(alg::HMCDA, ϵ)
    return AHMC.HMCKernel(
        AHMC.Trajectory{AHMC.EndPointTS}(AHMC.Leapfrog(ϵ), AHMC.FixedIntegrationTime(alg.λ))
    )
end
function make_ahmc_kernel(alg::NUTS, ϵ)
    return AHMC.HMCKernel(
        AHMC.Trajectory{AHMC.MultinomialTS}(
            AHMC.Leapfrog(ϵ), AHMC.GeneralisedNoUTurn(alg.max_depth, alg.Δ_max)
        ),
    )
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    rng, ::Sampler{<:Hamiltonian}, dist::Distribution, vn::VarName, vi
)
    return DynamicPPL.assume(dist, vn, vi)
end

####
#### Default HMC stepsize and mass matrix adaptor
####

function AHMCAdaptor(alg::AdaptiveHamiltonian, metric::AHMC.AbstractMetric; ϵ=alg.ϵ)
    pc = AHMC.MassMatrixAdaptor(metric)
    da = AHMC.StepSizeAdaptor(alg.δ, ϵ)

    if iszero(alg.n_adapts)
        adaptor = AHMC.Adaptation.NoAdaptation()
    else
        if metric == AHMC.UnitEuclideanMetric
            adaptor = AHMC.NaiveHMCAdaptor(pc, da)  # there is actually no adaptation for mass matrix
        else
            adaptor = AHMC.StanHMCAdaptor(pc, da)
            AHMC.initialize!(adaptor, alg.n_adapts)
        end
    end

    return adaptor
end

function AHMCAdaptor(::Hamiltonian, ::AHMC.AbstractMetric; kwargs...)
    return AHMC.Adaptation.NoAdaptation()
end
