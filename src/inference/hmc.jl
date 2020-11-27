###
### Sampler states
###

struct HMCState{
    TV<:AbstractVarInfo,
    TTraj<:AHMC.AbstractTrajectory,
    THam<:AHMC.Hamiltonian,
    PhType<:AHMC.PhasePoint,
    TAdapt<:AHMC.Adaptation.AbstractAdaptor,
}
    vi::TV
    i::Int
    traj::TTraj
    hamiltonian::THam
    z::PhType
    adaptor::TAdapt
end

# TODO: Include recompute Hamiltonian here?
function gibbs_update_state(state::HMCState, varinfo::AbstractVarInfo)
    return HMCState(varinfo, state.i, state.traj, state.hamiltonian, state.z, state.adaptor)
end

##########################
# Hamiltonian Transition #
##########################

struct HMCTransition{T,NT<:NamedTuple,F<:AbstractFloat}
    θ::T
    lp::F
    stat::NT
end

function HMCTransition(vi::AbstractVarInfo, t::AHMC.Transition)
    theta = tonamedtuple(vi)
    lp = getlogp(vi)
    return HMCTransition(theta, lp, t.stat)
end

function metadata(t::HMCTransition)
    return merge((lp = t.lp,), t.stat)
end

DynamicPPL.getlogp(t::HMCTransition) = t.lp

###
### Hamiltonian Monte Carlo samplers.
###

"""
    HMC(ϵ::Float64, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler with static trajectory.

Arguments:

- `ϵ::Float64` : The leapfrog step size to use.
- `n_leapfrog::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(0.05, 10)
```

Tips:

- If you are receiving gradient errors when using `HMC`, try reducing the leapfrog step size `ϵ`, e.g.

```julia
# Original step size
sample(gdemo([1.5, 2]), HMC(0.1, 10), 1000)

# Reduced step size
sample(gdemo([1.5, 2]), HMC(0.01, 10), 1000)
```
"""
struct HMC{AD, space, metricT <: AHMC.AbstractMetric} <: StaticHamiltonian{AD}
    ϵ::Float64 # leapfrog step size
    n_leapfrog::Int # leapfrog step number
end

isgibbscomponent(::Hamiltonian) = true

HMC(args...; kwargs...) = HMC{ADBackend()}(args...; kwargs...)
function HMC{AD}(ϵ::Float64, n_leapfrog::Int, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
    return HMC{AD, space, metricT}(ϵ, n_leapfrog)
end
function HMC{AD}(
    ϵ::Float64,
    n_leapfrog::Int,
    ::Tuple{};
    kwargs...
) where AD
    return HMC{AD}(ϵ, n_leapfrog; kwargs...)
end
function HMC{AD}(
    ϵ::Float64,
    n_leapfrog::Int,
    space::Symbol...;
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMC{AD}(ϵ, n_leapfrog, metricT, space)
end

DynamicPPL.initialsampler(::Sampler{<:Hamiltonian}) = SampleFromUniform()

# Handle setting `nadapts` and `discard_initial`
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Sampler{<:AdaptiveHamiltonian},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    nadapts=sampler.alg.n_adapts,
    discard_adapt=true,
    discard_initial=-1,
    kwargs...
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

        return AbstractMCMC.mcmcsample(rng, model, sampler, N;
                                       chain_type=chain_type, progress=progress,
                                       nadapts=_nadapts, discard_initial=_discard_initial,
                                       kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=progress,
                      nadapts=0, discard_adapt=false, discard_initial=0, kwargs...)
    end
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Hamiltonian},
    vi::AbstractVarInfo;
    init_params=nothing,
    nadapts=0,
    kwargs...
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    link!(vi, spl)
    model(rng, vi, spl)

    # Extract parameters.
    theta = vi[spl]

    # Create a Hamiltonian.
    metricT = getmetricT(spl.alg)
    metric = metricT(length(theta))
    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)
    hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

    # Compute phase point z.
    z = AHMC.phasepoint(rng, theta, hamiltonian)

    # If no initial parameters are provided, resample until the log probability
    # and its gradient are finite.
    if init_params === nothing
        while !isfinite(z)
            model(rng, vi, SampleFromUniform())
            link!(vi, spl)
            theta = vi[spl]

            hamiltonian = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
            z = AHMC.phasepoint(rng, theta, hamiltonian)
        end
    end

    # Cache current log density.
    log_density_old = getlogp(vi)

    # Find good eps if not provided one
    if iszero(spl.alg.ϵ)
        ϵ = AHMC.find_good_stepsize(hamiltonian, theta)
        @info "Found initial step size" ϵ
    else
        ϵ = spl.alg.ϵ
    end

    # Generate a trajectory.
    traj = gen_traj(spl.alg, ϵ)

    # Create initial transition and state.
    # Already perform one step since otherwise we don't get any statistics.
    t = AHMC.step(rng, hamiltonian, traj, z)

    # Adaptation
    adaptor = AHMCAdaptor(spl.alg, hamiltonian.metric; ϵ=ϵ)
    if spl.alg isa AdaptiveHamiltonian
        hamiltonian, traj, _ =
            AHMC.adapt!(hamiltonian, traj, adaptor,
                        1, nadapts, t.z.θ, t.stat.acceptance_rate)
    end

    # Update `vi` based on acceptance
    @debug "decide whether to accept..."
    if t.stat.is_accept
        vi[spl] = t.z.θ
        setlogp!(vi, t.stat.log_density)
    else
        vi[spl] = theta
        setlogp!(vi, log_density_old)
    end

    transition = HMCTransition(vi, t)
    state = HMCState(vi, 1, traj, hamiltonian, t.z, adaptor)

    # If a Gibbs component, transform the values back to the constrained space.
    if spl.selector.tag !== :default
        @debug "R -> X..."
        invlink!(vi, spl)
    end

    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    state::HMCState;
    nadapts=0,
    kwargs...
)
    # Get step size
    ϵ = getstepsize(spl, state)
    @debug "current ϵ" ϵ

    # Get VarInfo object
    vi = state.vi
    i = state.i + 1

    # When a Gibbs component, transform values to the unconstrained space.
    if spl.selector.tag !== :default
        @debug "X-> R..."
        link!(vi, spl)
        model(rng, vi, spl)
    end

    # Get position and log density before transition
    θ_old = vi[spl]
    log_density_old = getlogp(vi)
    hamiltonian = if spl.selector.tag === :default
        state.hamiltonian
    else
        get_hamiltonian(model, spl, vi, state, length(θ_old))
    end

    z = if spl.selector.tag === :default
        state.z
    else
        resize!(state.z.θ, length(θ_old))
        state.z.θ .= θ_old
        state.z
    end

    # Compute transition.
    t = AHMC.step(rng, hamiltonian, state.traj, z)

    # Adaptation
    if spl.alg isa AdaptiveHamiltonian
        hamiltonian, traj, _ =
            AHMC.adapt!(hamiltonian, state.traj, state.adaptor,
                        i, nadapts, t.z.θ, t.stat.acceptance_rate)
    else
        traj = state.traj
    end

    # Update `vi` based on acceptance
    @debug "decide whether to accept..."
    if t.stat.is_accept
        vi[spl] = t.z.θ
        setlogp!(vi, t.stat.log_density)
    else
        vi[spl] = θ_old
        setlogp!(vi, log_density_old)
    end

    # Compute next transition and state.
    transition = HMCTransition(vi, t)
    newstate = HMCState(vi, i, traj, hamiltonian, t.z, state.adaptor)

    # If a Gibbs component, transform the values back to the constrained space.
    if spl.selector.tag !== :default
        @debug "R -> X..."
        invlink!(vi, spl)
    end

    return transition, newstate
end

function get_hamiltonian(model, spl, vi, state, n)
    metric = gen_metric(n, spl, state)
    ℓπ = gen_logπ(vi, spl, model)
    ∂ℓπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    return AHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
end

"""
    HMCDA(n_adapts::Int, δ::Float64, λ::Float64; ϵ::Float64=0.0)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(200, 0.65, 0.3)
```

Arguments:

- `n_adapts::Int` : Numbers of samples to use for adaptation.
- `δ::Float64` : Target acceptance rate. 65% is often recommended.
- `λ::Float64` : Target leapfrop length.
- `ϵ::Float64=0.0` : Inital step size; 0 means automatically search by Turing.

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
struct HMCDA{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_adapts    ::  Int         # number of samples with adaption for ϵ
    δ           ::  Float64     # target accept rate
    λ           ::  Float64     # target leapfrog length
    ϵ           ::  Float64     # (initial) step size
end
HMCDA(args...; kwargs...) = HMCDA{ADBackend()}(args...; kwargs...)
function HMCDA{AD}(n_adapts::Int, δ::Float64, λ::Float64, ϵ::Float64, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
    return HMCDA{AD, space, metricT}(n_adapts, δ, λ, ϵ)
end

function HMCDA{AD}(
    δ::Float64,
    λ::Float64;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMCDA{AD}(-1, δ, λ, init_ϵ, metricT, ())
end

function HMCDA{AD}(
    n_adapts::Int,
    δ::Float64,
    λ::Float64,
    ::Tuple{};
    kwargs...
) where AD
    return HMCDA{AD}(n_adapts, δ, λ; kwargs...)
end

function HMCDA{AD}(
    n_adapts::Int,
    δ::Float64,
    λ::Float64,
    space::Symbol...;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMCDA{AD}(n_adapts, δ, λ, init_ϵ, metricT, space)
end


"""
    NUTS(n_adapts::Int, δ::Float64; max_depth::Int=5, Δ_max::Float64=1000.0, ϵ::Float64=0.0)

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
- `init_ϵ::Float64` : Inital step size; 0 means automatically searching using a heuristic procedure.

"""
struct NUTS{AD,space,metricT<:AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_adapts::Int         # number of samples with adaption for ϵ
    δ::Float64        # target accept rate
    max_depth::Int         # maximum tree depth
    Δ_max::Float64
    ϵ::Float64     # (initial) step size
end

NUTS(args...; kwargs...) = NUTS{ADBackend()}(args...; kwargs...)

function NUTS{AD}(
    n_adapts::Int,
    δ::Float64,
    max_depth::Int,
    Δ_max::Float64,
    ϵ::Float64,
    ::Type{metricT},
    space::Tuple
) where {AD, metricT}
    return NUTS{AD, space, metricT}(n_adapts, δ, max_depth, Δ_max, ϵ)
end

function NUTS{AD}(
    n_adapts::Int,
    δ::Float64,
    ::Tuple{};
    kwargs...
) where AD
    NUTS{AD}(n_adapts, δ; kwargs...)
end

function NUTS{AD}(
    n_adapts::Int,
    δ::Float64,
    space::Symbol...;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric
) where AD
    NUTS{AD}(n_adapts, δ, max_depth, Δ_max, init_ϵ, metricT, space)
end

function NUTS{AD}(
    δ::Float64;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric
) where AD
    NUTS{AD}(-1, δ, max_depth, Δ_max, init_ϵ, metricT, ())
end

function NUTS{AD}(kwargs...) where AD
    NUTS{AD}(-1, 0.65; kwargs...)
end

for alg in (:HMC, :HMCDA, :NUTS)
    @eval getmetricT(::$alg{<:Any, <:Any, metricT}) where {metricT} = metricT
end

#####
##### HMC core functions
#####

getstepsize(sampler::Sampler{<:Hamiltonian}, state) = sampler.alg.ϵ
getstepsize(sampler::Sampler{<:AdaptiveHamiltonian}, state) = AHMC.getϵ(state.adaptor)

"""
    gen_∂logπ∂θ(vi, spl::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, spl, model)`.
"""
function gen_∂logπ∂θ(vi, spl::Sampler, model)
    function ∂logπ∂θ(x)
        return gradient_logp(x, vi, model, spl)
    end
    return ∂logπ∂θ
end

"""
    gen_logπ(vi, spl::Sampler, model)

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, spl, model)`.
"""
function gen_logπ(vi, spl::AbstractSampler, model)
    function logπ(x)::Float64
        x_old, lj_old = vi[spl], getlogp(vi)
        vi[spl] = x
        model(vi, spl)
        lj = getlogp(vi)
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

gen_metric(dim::Int, spl::Sampler{<:Hamiltonian}, state) = AHMC.UnitEuclideanMetric(dim)
function gen_metric(dim::Int, spl::Sampler{<:AdaptiveHamiltonian}, state)
    return AHMC.renew(state.hamiltonian.metric, AHMC.getM⁻¹(state.adaptor.pc))
end

gen_traj(alg::HMC, ϵ) = AHMC.StaticTrajectory(AHMC.Leapfrog(ϵ), alg.n_leapfrog)
gen_traj(alg::HMCDA, ϵ) = AHMC.HMCDA(AHMC.Leapfrog(ϵ), alg.λ)
gen_traj(alg::NUTS, ϵ) = AHMC.NUTS(AHMC.Leapfrog(ϵ), alg.max_depth, alg.Δ_max)

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    rng,
    spl::Sampler{<:Hamiltonian},
    dist::Distribution,
    vn::VarName,
    vi,
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:Hamiltonian},
    dist::MultivariateDistribution,
    vns::AbstractArray{<:VarName},
    var::AbstractMatrix,
    vi,
)
    @assert length(dist) == size(var, 1)
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:Hamiltonian},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi,
)
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function DynamicPPL.observe(
    spl::Sampler{<:Hamiltonian},
    d::Distribution,
    value,
    vi,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:Hamiltonian},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
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

AHMCAdaptor(::Hamiltonian, ::AHMC.AbstractMetric; kwargs...) = AHMC.Adaptation.NoAdaptation()

##########################
# HMC State Constructors #
##########################

function HMCState(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    vi::AbstractVarInfo;
    kwargs...
)
    # Link everything if needed.
    !islinked(vi, spl) && link!(vi, spl)

    # Get the initial log pdf and gradient functions.
    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)

    # Get the metric type.
    metricT = getmetricT(spl.alg)

    # Create a Hamiltonian.
    θ_init = Vector{Float64}(spl.state.vi[spl])
    metric = metricT(length(θ_init))
    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

    # Find good eps if not provided one
    if iszero(spl.alg.ϵ)
        ϵ = AHMC.find_good_stepsize(h, θ_init)
        @info "Found initial step size" ϵ
    else
        ϵ = spl.alg.ϵ
    end

    # Generate a trajectory.
    traj = gen_traj(spl.alg, ϵ)

    # Generate a phasepoint. Replaced during sample_init!
    h, t = AHMC.sample_init(rng, h, θ_init) # this also ensure AHMC has the same dim as θ.

    # Unlink everything.
    invlink!(vi, spl)

    return HMCState(vi, 0, 0, traj, h, AHMCAdaptor(spl.alg, metric; ϵ=ϵ), t.z)
end
