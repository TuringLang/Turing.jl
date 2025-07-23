abstract type Hamiltonian <: InferenceAlgorithm end
abstract type StaticHamiltonian <: Hamiltonian end
abstract type AdaptiveHamiltonian <: Hamiltonian end

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
    i::Int
    kernel::TKernel
    hamiltonian::THam
    z::PhType
    adaptor::TAdapt
end

###
### Hamiltonian Monte Carlo samplers.
###

varinfo(state::HMCState) = state.vi

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

# Handle setting `nadapts` and `discard_initial`
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Sampler{<:AdaptiveHamiltonian},
    N::Integer;
    chain_type=DynamicPPL.default_chain_type(sampler),
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    progress=PROGRESS[],
    nadapts=sampler.alg.n_adapts,
    discard_adapt=true,
    discard_initial=-1,
    kwargs...,
)
    if resume_from === nothing
        # If `nadapts` is `-1`, then the user called a convenience
        # constructor like `NUTS()` or `NUTS(0.65)`,
        # and we should set a default for them.
        if nadapts == -1
            _nadapts = min(1000, N ÷ 2)
        else
            _nadapts = nadapts
        end

        # If `discard_initial` is `-1`, then users did not specify the keyword argument.
        if discard_initial == -1
            _discard_initial = discard_adapt ? _nadapts : 0
        else
            _discard_initial = discard_initial
        end

        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type=chain_type,
            progress=progress,
            nadapts=_nadapts,
            discard_initial=_discard_initial,
            kwargs...,
        )
    else
        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type=chain_type,
            initial_state=initial_state,
            progress=progress,
            nadapts=0,
            discard_adapt=false,
            discard_initial=0,
            kwargs...,
        )
    end
end

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
        "failed to find valid initial parameters in $(max_attempts) tries. This may indicate an error with the model or AD backend; please open an issue at https://github.com/TuringLang/Turing.jl/issues",
    )
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Hamiltonian},
    vi_original::AbstractVarInfo;
    initial_params=nothing,
    nadapts=0,
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
        model, DynamicPPL.getlogjoint, vi; adtype=spl.alg.adtype
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

    # Cache current log density. We will reuse this if the transition is rejected.
    logp_old = DynamicPPL.getlogp(vi)

    # Find good eps if not provided one
    if iszero(spl.alg.ϵ)
        ϵ = AHMC.find_good_stepsize(rng, hamiltonian, theta)
        @info "Found initial step size" ϵ
    else
        ϵ = spl.alg.ϵ
    end

    # Generate a kernel.
    kernel = make_ahmc_kernel(spl.alg, ϵ)

    # Create initial transition and state.
    # Already perform one step since otherwise we don't get any statistics.
    t = AHMC.transition(rng, hamiltonian, kernel, z)

    # Adaptation
    adaptor = AHMCAdaptor(spl.alg, hamiltonian.metric; ϵ=ϵ)
    if spl.alg isa AdaptiveHamiltonian
        hamiltonian, kernel, _ = AHMC.adapt!(
            hamiltonian, kernel, adaptor, 1, nadapts, t.z.θ, t.stat.acceptance_rate
        )
    end

    # Update VarInfo based on acceptance
    if t.stat.is_accept
        vi = DynamicPPL.unflatten(vi, t.z.θ)
        # Re-evaluate to calculate log probability density.
        # TODO(penelopeysm): This seems a little bit wasteful. Unfortunately,
        # even though `t.stat.log_density` contains some kind of logp, this
        # doesn't track prior and likelihood separately but rather a single
        # log-joint (and in linked space), so which we have no way to decompose
        # this back into prior and likelihood. I don't immediately see how to
        # solve this without re-evaluating the model.
        _, vi = DynamicPPL.evaluate!!(model, vi)
    else
        # Reset VarInfo back to its original state.
        vi = DynamicPPL.unflatten(vi, theta)
        vi = DynamicPPL.setlogp!!(vi, logp_old)
    end

    transition = Transition(model, vi, t)
    state = HMCState(vi, 1, kernel, hamiltonian, t.z, adaptor)

    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    state::HMCState;
    nadapts=0,
    kwargs...,
)
    # Get step size
    @debug "current ϵ" getstepsize(spl, state)

    # Compute transition.
    hamiltonian = state.hamiltonian
    z = state.z
    t = AHMC.transition(rng, hamiltonian, state.kernel, z)

    # Adaptation
    i = state.i + 1
    if spl.alg isa AdaptiveHamiltonian
        hamiltonian, kernel, _ = AHMC.adapt!(
            hamiltonian,
            state.kernel,
            state.adaptor,
            i,
            nadapts,
            t.z.θ,
            t.stat.acceptance_rate,
        )
    else
        kernel = state.kernel
    end

    # Update variables
    vi = state.vi
    if t.stat.is_accept
        vi = DynamicPPL.unflatten(vi, t.z.θ)
        # Re-evaluate to calculate log probability density.
        # TODO(penelopeysm): This seems a little bit wasteful. See note above.
        _, vi = DynamicPPL.evaluate!!(model, vi)
    end

    # Compute next transition and state.
    transition = Transition(model, vi, t)
    newstate = HMCState(vi, i, kernel, hamiltonian, t.z, state.adaptor)

    return transition, newstate
end

function get_hamiltonian(model, spl, vi, state, n)
    metric = gen_metric(n, spl, state)
    ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint, vi; adtype=spl.alg.adtype
    )
    lp_func = Base.Fix1(LogDensityProblems.logdensity, ldf)
    lp_grad_func = Base.Fix1(LogDensityProblems.logdensity_and_gradient, ldf)
    return AHMC.Hamiltonian(metric, lp_func, lp_grad_func)
end

"""
    HMCDA(
        n_adapts::Int, δ::Float64, λ::Float64; ϵ::Float64 = 0.0;
        adtype::ADTypes.AbstractADType = AutoForwardDiff(),
    )

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

# Usage

```julia
HMCDA(200, 0.65, 0.3)
```

# Arguments

- `n_adapts`: Numbers of samples to use for adaptation.
- `δ`: Target acceptance rate. 65% is often recommended.
- `λ`: Target leapfrog length.
- `ϵ`: Initial step size; 0 means automatically search by Turing.
- `adtype`: The automatic differentiation (AD) backend.
    If not specified, `ForwardDiff` is used, with its `chunksize` automatically determined.

# Reference

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
Research 15, no. 1 (2014): 1593-1623.
"""
struct HMCDA{AD,metricT<:AHMC.AbstractMetric} <: AdaptiveHamiltonian
    n_adapts::Int         # number of samples with adaption for ϵ
    δ::Float64     # target accept rate
    λ::Float64     # target leapfrog length
    ϵ::Float64     # (initial) step size
    adtype::AD
end

function HMCDA(
    n_adapts::Int,
    δ::Float64,
    λ::Float64,
    ϵ::Float64,
    ::Type{metricT};
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
) where {metricT<:AHMC.AbstractMetric}
    return HMCDA{typeof(adtype),metricT}(n_adapts, δ, λ, ϵ, adtype)
end

function HMCDA(
    δ::Float64,
    λ::Float64;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    return HMCDA(-1, δ, λ, init_ϵ, metricT; adtype=adtype)
end

function HMCDA(n_adapts::Int, δ::Float64, λ::Float64, ::Tuple{}; kwargs...)
    return HMCDA(n_adapts, δ, λ; kwargs...)
end

function HMCDA(
    n_adapts::Int,
    δ::Float64,
    λ::Float64;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    return HMCDA(n_adapts, δ, λ, init_ϵ, metricT; adtype=adtype)
end

"""
    NUTS(n_adapts::Int, δ::Float64; max_depth::Int=10, Δ_max::Float64=1000.0, init_ϵ::Float64=0.0; adtype::ADTypes.AbstractADType=AutoForwardDiff()

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS()            # Use default NUTS configuration.
NUTS(1000, 0.65)  # Use 1000 adaption steps, and target accept ratio 0.65.
```

Arguments:

- `n_adapts::Int` : The number of samples to use with adaptation.
- `δ::Float64` : Target acceptance rate for dual averaging.
- `max_depth::Int` : Maximum doubling tree depth.
- `Δ_max::Float64` : Maximum divergence during doubling tree.
- `init_ϵ::Float64` : Initial step size; 0 means automatically searching using a heuristic procedure.
- `adtype::ADTypes.AbstractADType` : The automatic differentiation (AD) backend.
    If not specified, `ForwardDiff` is used, with its `chunksize` automatically determined.

"""
struct NUTS{AD,metricT<:AHMC.AbstractMetric} <: AdaptiveHamiltonian
    n_adapts::Int         # number of samples with adaption for ϵ
    δ::Float64        # target accept rate
    max_depth::Int         # maximum tree depth
    Δ_max::Float64
    ϵ::Float64     # (initial) step size
    adtype::AD
end

function NUTS(
    n_adapts::Int,
    δ::Float64,
    max_depth::Int,
    Δ_max::Float64,
    ϵ::Float64,
    ::Type{metricT};
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
) where {metricT}
    return NUTS{typeof(adtype),metricT}(n_adapts, δ, max_depth, Δ_max, ϵ, adtype)
end

function NUTS(n_adapts::Int, δ::Float64, ::Tuple{}; kwargs...)
    return NUTS(n_adapts, δ; kwargs...)
end

function NUTS(
    n_adapts::Int,
    δ::Float64;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    return NUTS(n_adapts, δ, max_depth, Δ_max, init_ϵ, metricT; adtype=adtype)
end

function NUTS(
    δ::Float64;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    return NUTS(-1, δ, max_depth, Δ_max, init_ϵ, metricT; adtype=adtype)
end

function NUTS(; kwargs...)
    return NUTS(-1, 0.65; kwargs...)
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
