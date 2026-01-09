abstract type Hamiltonian <: AbstractSampler end
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
    L<:DynamicPPL.LogDensityFunction,
}
    vi::TV
    i::Int
    kernel::TKernel
    hamiltonian::THam
    z::PhType
    adaptor::TAdapt
    ldf::L
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

Turing.Inference.init_strategy(::Hamiltonian) = DynamicPPL.InitFromUniform()

# Handle setting `nadapts` and `discard_initial`
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::AdaptiveHamiltonian,
    N::Integer;
    check_model=true,
    chain_type=DEFAULT_CHAIN_TYPE,
    initial_params=Turing.Inference.init_strategy(sampler),
    initial_state=nothing,
    progress=PROGRESS[],
    nadapts=sampler.n_adapts,
    discard_adapt=true,
    discard_initial=-1,
    kwargs...,
)
    check_model && _check_model(model, sampler)
    if initial_state === nothing
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
            initial_state=initial_state,
            progress=progress,
            nadapts=_nadapts,
            discard_initial=_discard_initial,
            initial_params=initial_params,
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
            initial_params=initial_params,
            kwargs...,
        )
    end
end

function find_initial_params(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo,
    hamiltonian::AHMC.Hamiltonian,
    init_strategy::DynamicPPL.AbstractInitStrategy;
    max_attempts::Int=1000,
)
    varinfo = deepcopy(varinfo)  # Don't mutate

    for attempts in 1:max_attempts
        theta = varinfo[:]
        z = AHMC.phasepoint(rng, theta, hamiltonian)
        isfinite(z) && return varinfo, z

        attempts == 10 &&
            @warn "failed to find valid initial parameters in $(attempts) tries; consider providing a different initialisation strategy with the `initial_params` keyword"

        # Resample and try again.
        _, varinfo = DynamicPPL.init!!(rng, model, varinfo, init_strategy)
    end

    # if we failed to find valid initial parameters, error
    return error(
        "failed to find valid initial parameters in $(max_attempts) tries. See https://turinglang.org/docs/uri/initial-parameters for common causes and solutions. If the issue persists, please open an issue at https://github.com/TuringLang/Turing.jl/issues",
    )
end

function Turing.Inference.initialstep(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Hamiltonian,
    vi_original::AbstractVarInfo;
    # the initial_params kwarg is always passed on from sample(), cf. DynamicPPL
    # src/sampler.jl, so we don't need to provide a default value here
    initial_params::DynamicPPL.AbstractInitStrategy,
    nadapts=0,
    verbose::Bool=true,
    kwargs...,
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    vi = DynamicPPL.link(vi_original, model)

    # Extract parameters.
    theta = vi[:]

    # Create a Hamiltonian.
    metricT = getmetricT(spl)
    metric = metricT(length(theta))
    ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.adtype
    )
    lp_func = Base.Fix1(LogDensityProblems.logdensity, ldf)
    lp_grad_func = Base.Fix1(LogDensityProblems.logdensity_and_gradient, ldf)
    hamiltonian = AHMC.Hamiltonian(metric, lp_func, lp_grad_func)

    # Note that there is already one round of 'initialisation' before we reach this step,
    # inside DynamicPPL's `AbstractMCMC.step` implementation. That leads to a possible issue
    # that this `find_initial_params` function might override the parameters set by the
    # user.
    # Luckily for us, `find_initial_params` always checks if the logp and its gradient are
    # finite. If it is already finite with the params inside the current `vi`, it doesn't
    # attempt to find new ones. This means that the parameters passed to `sample()` will be
    # respected instead of being overridden here.
    vi, z = find_initial_params(rng, model, vi, hamiltonian, initial_params)
    theta = vi[:]

    # Find good eps if not provided one
    if iszero(spl.ϵ)
        ϵ = AHMC.find_good_stepsize(rng, hamiltonian, theta)
        verbose && @info "Found initial step size" ϵ
    else
        ϵ = spl.ϵ
    end
    # Generate a kernel and adaptor.
    kernel = make_ahmc_kernel(spl, ϵ)
    adaptor = AHMCAdaptor(spl, hamiltonian.metric, nadapts; ϵ=ϵ)

    transition = DynamicPPL.ParamsWithStats(theta, ldf, NamedTuple())
    state = HMCState(vi, 0, kernel, hamiltonian, z, adaptor, ldf)

    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Hamiltonian,
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
    if spl isa AdaptiveHamiltonian
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
    end

    # Compute next transition and state.
    transition = DynamicPPL.ParamsWithStats(t.z.θ, state.ldf, t.stat)
    newstate = HMCState(vi, i, kernel, hamiltonian, t.z, state.adaptor, state.ldf)

    return transition, newstate
end

function get_hamiltonian(model, spl, vi, state, n)
    metric = gen_metric(n, spl, state)
    ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.adtype
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

getstepsize(sampler::Hamiltonian, state) = sampler.ϵ
getstepsize(sampler::AdaptiveHamiltonian, state) = AHMC.getϵ(state.adaptor)
function getstepsize(
    sampler::AdaptiveHamiltonian,
    state::HMCState{TV,TKernel,THam,PhType,AHMC.Adaptation.NoAdaptation},
) where {TV,TKernel,THam,PhType}
    return state.kernel.τ.integrator.ϵ
end

gen_metric(dim::Int, spl::Hamiltonian, state) = AHMC.UnitEuclideanMetric(dim)
function gen_metric(dim::Int, spl::AdaptiveHamiltonian, state)
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
#### Default HMC stepsize and mass matrix adaptor
####

function AHMCAdaptor(
    alg::AdaptiveHamiltonian, metric::AHMC.AbstractMetric, nadapts::Int; ϵ=alg.ϵ
)
    pc = AHMC.MassMatrixAdaptor(metric)
    da = AHMC.StepSizeAdaptor(alg.δ, ϵ)

    if iszero(alg.n_adapts)
        adaptor = AHMC.Adaptation.NoAdaptation()
    else
        if metric == AHMC.UnitEuclideanMetric
            adaptor = AHMC.NaiveHMCAdaptor(pc, da)  # there is actually no adaptation for mass matrix
        else
            adaptor = AHMC.StanHMCAdaptor(pc, da)
            AHMC.initialize!(adaptor, nadapts)
        end
    end

    return adaptor
end

function AHMCAdaptor(::Hamiltonian, ::AHMC.AbstractMetric, nadapts::Int; kwargs...)
    return AHMC.Adaptation.NoAdaptation()
end
