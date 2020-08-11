###
### Sampler states
###

mutable struct HMCState{
    TV <: TypedVarInfo,
    TTraj<:AHMC.AbstractTrajectory,
    TAdapt<:AHMC.Adaptation.AbstractAdaptor,
    PhType <: AHMC.PhasePoint
} <: AbstractSamplerState
    vi       :: TV
    eval_num :: Int
    i        :: Int
    traj     :: TTraj
    h        :: AHMC.Hamiltonian
    adaptor  :: TAdapt
    z        :: PhType
end

##########################
# Hamiltonian Transition #
##########################

struct HamiltonianTransition{T, NT<:NamedTuple, F<:AbstractFloat}
    θ    :: T
    lp   :: F
    stat :: NT
end

function HamiltonianTransition(spl::Sampler{<:Hamiltonian}, t::AHMC.Transition)
    theta = tonamedtuple(spl.state.vi)
    lp = getlogp(spl.state.vi)
    return HamiltonianTransition(theta, lp, t.stat)
end

function additional_parameters(::Type{<:HamiltonianTransition})
    return [:lp,:stat]
end

DynamicPPL.getlogp(t::HamiltonianTransition) = t.lp

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
mutable struct HMC{AD, space, metricT <: AHMC.AbstractMetric} <: StaticHamiltonian{AD}
    ϵ           ::  Float64   # leapfrog step size
    n_leapfrog  ::  Int       # leapfrog step number
end

DynamicPPL.alg_str(::Sampler{<:Hamiltonian}) = "HMC"
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

function update_hamiltonian!(spl, model, n)
    metric = gen_metric(n, spl)
    ℓπ = gen_logπ(spl.state.vi, spl, model)
    ∂ℓπ∂θ = gen_∂logπ∂θ(spl.state.vi, spl, model)
    spl.state.h = AHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    return spl
end

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Hamiltonian},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    init_theta=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; init_theta=init_theta, verbose=verbose, kwargs...)
    if init_theta !== nothing
        # Doesn't support dynamic models
        link!(spl.state.vi, spl)
        model(spl.state.vi, spl)
        theta = spl.state.vi[spl]
        update_hamiltonian!(spl, model, length(theta))
        # Refresh the internal cache phase point z's hamiltonian energy.
        spl.state.z = AHMC.phasepoint(rng, theta, spl.state.h)
    else
        # Samples new values and sets trans to true, then computes the logp
        model(empty!(spl.state.vi), SampleFromUniform())
        link!(spl.state.vi, spl)
        theta = spl.state.vi[spl]
        update_hamiltonian!(spl, model, length(theta))
        # Refresh the internal cache phase point z's hamiltonian energy.
        spl.state.z = AHMC.phasepoint(rng, theta, spl.state.h)
        while !isfinite(spl.state.z.ℓπ.value) || !isfinite(spl.state.z.ℓπ.gradient)
            model(empty!(spl.state.vi), SampleFromUniform())
            link!(spl.state.vi, spl)
            theta = spl.state.vi[spl]
            update_hamiltonian!(spl, model, length(theta))
            # Refresh the internal cache phase point z's hamiltonian energy.
            spl.state.z = AHMC.phasepoint(rng, theta, spl.state.h)
        end
    end

    # Set the default number of adaptations, if relevant.
    if spl.alg isa AdaptiveHamiltonian
        # If there's no chain passed in, verify the n_adapts.
        if resume_from === nothing
            # if n_adapts is -1, then the user called a convenience
            # constructor like NUTS() or NUTS(0.65), and we should
            # set a default for them.
            if spl.alg.n_adapts == -1
                spl.alg.n_adapts = min(1000, N ÷ 2)
            elseif spl.alg.n_adapts > N
                # Verify that n_adapts is not greater than the number of samples to draw.
                throw(ArgumentError("n_adapt of $(spl.alg.n_adapts) is greater than total samples of $N."))
            end
        else
            spl.alg.n_adapts = 0
        end
    end

    # Convert to transformed space if we're using
    # non-Gibbs sampling.
    if !islinked(spl.state.vi, spl) && spl.selector.tag == :default
        link!(spl.state.vi, spl)
        model(spl.state.vi, spl)
    elseif islinked(spl.state.vi, spl) && spl.selector.tag != :default
        invlink!(spl.state.vi, spl)
        model(spl.state.vi, spl)        
    end
end

function AbstractMCMC.transitions_init(
    transition,
    ::AbstractModel,
    sampler::Sampler{<:Hamiltonian},
    N::Integer;
    discard_adapt = true,
    kwargs...
)
    if discard_adapt && isdefined(sampler.alg, :n_adapts)
        n = max(0, N - sampler.alg.n_adapts)
    else
        n = N
    end
    return Vector{typeof(transition)}(undef, n)
end

function AbstractMCMC.transitions_save!(
    transitions::AbstractVector,
    iteration::Integer,
    transition,
    ::AbstractModel,
    sampler::Sampler{<:Hamiltonian},
    ::Integer;
    discard_adapt = true,
    kwargs...
)
    if discard_adapt && isdefined(sampler.alg, :n_adapts)
        if iteration > sampler.alg.n_adapts
            transitions[iteration - sampler.alg.n_adapts] = transition
        end
        return
    end

    transitions[iteration] = transition
    return
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
mutable struct HMCDA{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
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
- `max_depth::Float64` : Maximum doubling tree depth.
- `Δ_max::Float64` : Maximum divergence during doubling tree.
- `ϵ::Float64` : Inital step size; 0 means automatically searching using a heuristic procedure.

"""
mutable struct NUTS{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_adapts    ::  Int         # number of samples with adaption for ϵ
    δ           ::  Float64     # target accept rate
    max_depth   ::  Int         # maximum tree depth
    Δ_max       ::  Float64
    ϵ           ::  Float64     # (initial) step size
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

####
#### Sampler construction
####

# Sampler(alg::Hamiltonian) =  Sampler(alg, AHMCAdaptor())
function Sampler(
    alg::Union{StaticHamiltonian, AdaptiveHamiltonian},
    model::Model,
    s::Selector=Selector()
)
    info = Dict{Symbol, Any}()
    # Create an empty sampler state that just holds a typed VarInfo.
    initial_state = SamplerState(VarInfo(model))

    # Create an initial sampler, to get all the initialization out of the way.
    initial_spl = Sampler(alg, info, s, initial_state)

    # Create the actual state based on the alg type.
    state = HMCState(model, initial_spl, Random.GLOBAL_RNG)

    # Create a real sampler after getting all the types/running the init phase.
    return Sampler(alg, initial_spl.info, initial_spl.selector, state)
end



####
#### Transition / step functions for HMC samplers.
####

# Single step of a Hamiltonian.
function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    N::Integer,
    transition;
    kwargs...
)
    # Get step size
    ϵ = spl.alg isa AdaptiveHamiltonian ?
        AHMC.getϵ(spl.state.adaptor) :
        spl.alg.ϵ

    spl.state.i += 1
    spl.state.eval_num = 0

    Turing.DEBUG && @debug "current ϵ: $ϵ"

    # When a Gibbs component
    if spl.selector.tag != :default
        # Transform the space
        Turing.DEBUG && @debug "X-> R..."
        link!(spl.state.vi, spl)
        model(spl.state.vi, spl)
    end
    # Get position and log density before transition
    θ_old, log_density_old = spl.state.vi[spl], getlogp(spl.state.vi)
    if spl.selector.tag != :default
        update_hamiltonian!(spl, model, length(θ_old))
        resize!(spl.state.z.θ, length(θ_old))
        spl.state.z.θ .= θ_old
    end

    # Transition
    t = AHMC.step(rng, spl.state.h, spl.state.traj, spl.state.z)
    # Update z in state
    spl.state.z = t.z

    # Adaptation
    if spl.alg isa AdaptiveHamiltonian
        spl.state.h, spl.state.traj, isadapted = 
            AHMC.adapt!(spl.state.h, spl.state.traj, spl.state.adaptor, 
                        spl.state.i, spl.alg.n_adapts, t.z.θ, t.stat.acceptance_rate)
    end

    Turing.DEBUG && @debug "decide whether to accept..."

    # Update `vi` based on acceptance
    if t.stat.is_accept
        spl.state.vi[spl] = t.z.θ
        setlogp!(spl.state.vi, t.stat.log_density)
    else
        spl.state.vi[spl] = θ_old
        setlogp!(spl.state.vi, log_density_old)
    end

    # Gibbs component specified cares
    # Transform the space back
    Turing.DEBUG && @debug "R -> X..."
    spl.selector.tag != :default && invlink!(spl.state.vi, spl)

    return HamiltonianTransition(spl, t)
end


#####
##### HMC core functions
#####

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

gen_metric(dim::Int, spl::Sampler{<:Hamiltonian}) = AHMC.UnitEuclideanMetric(dim)
gen_metric(dim::Int, spl::Sampler{<:AdaptiveHamiltonian}) = AHMC.renew(spl.state.h.metric, AHMC.getM⁻¹(spl.state.adaptor.pc))

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
    Turing.DEBUG && _debug("assuming...")
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && _debug("dist = $dist")
    Turing.DEBUG && _debug("vn = $vn")
    Turing.DEBUG && _debug("r = $r, typeof(r)=$(typeof(r))")
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
    model::Model,
    spl::Sampler{<:Hamiltonian},
    rng::AbstractRNG;
    kwargs...
)

    # Reuse the VarInfo.
    vi = spl.state.vi

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
    if spl.alg.ϵ == 0.0
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
