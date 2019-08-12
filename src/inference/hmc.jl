###
### Sampler states
###

mutable struct StaticHMCState{
    TTraj<:AHMC.StaticTrajectory,
    TV <: TypedVarInfo
} <: SamplerState
    vi       :: TV
    eval_num :: Int
    i        :: Int
    traj     :: TTraj
    h        :: AHMC.Hamiltonian
end

mutable struct DynamicHMCState{
    TTraj<:AHMC.DynamicTrajectory,
    TAdapt<:AHMC.Adaptation.AbstractAdaptor,
    TV <: TypedVarInfo
} <: SamplerState
    vi       :: TV
    eval_num :: Int
    i        :: Int
    traj     :: TTraj
    h        :: AHMC.Hamiltonian
    adaptor  :: TAdapt
end


###
### Hamiltonian Monte Carlo samplers.
###

"""
    HMC(ϵ::Float64, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler.

Arguments:

- `ϵ::Float64` : The leapfrog step size to use.
- `n_leapfrog::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(0.05, 10)
```

Tips:

- If you are receiving gradient errors when using `HMC`, try reducing the
`step_size` parameter.

```julia
# Original step_size
sample(gdemo([1.5, 2]), HMC(1000, 0.1, 10))

# Reduced step_size.
sample(gdemo([1.5, 2]), HMC(1000, 0.01, 10))
```
"""
mutable struct HMC{AD, space, metricT <: AHMC.AbstractMetric} <: StaticHamiltonian{AD}
    ϵ           ::  Float64   # leapfrog step size
    n_leapfrog  ::  Int       # leapfrog step number
end

transition_type(::Sampler{<:Hamiltonian}) = Transition
alg_str(::Sampler{<:Hamiltonian}) = "HMC"

HMC(args...) = HMC{ADBackend()}(args...)
function HMC{AD}( ϵ::Float64, n_leapfrog::Int, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
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

function sample_init!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:Hamiltonian},
    N::Integer;
    adaptor=AHMCAdaptor(spl.alg),
    init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
    rng::AbstractRNG=GLOBAL_RNG,
    discard_adapt::Bool=true,
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; kwargs...)

    # Get `init_theta`
    if init_theta != nothing
        verbose && @info "Using passed-in initial variable values" init_theta
        # Convert individual numbers to length 1 vector; `ismissing(v)` is needed as `size(missing)` is undefined`
        init_theta = [ismissing(v) || size(v) == () ? [v] : v for v in init_theta]
        # Flatten `init_theta`
        init_theta_flat = foldl(vcat, map(vec, init_theta))
        # Create a mask to inidicate which values are not missing
        theta_mask = map(x -> !ismissing(x), init_theta_flat)
        # Get all values
        theta = vi[spl]
        @assert length(theta) == length(init_theta_flat) "Provided initial value doesn't match the dimension of the model"
        # Update those which are provided (i.e. not missing)
        theta[theta_mask] .= init_theta_flat[theta_mask]
        # Update in `vi`
        spl.state.vi[spl] = theta
    end

    # Set the defualt number of adaptations, if relevant.
    if spl.alg isa AdaptiveHamiltonian
        # If there's no chain passed in, verify the n_adapts.
        if resume_from === nothing
            if spl.alg.n_adapts == 0
                n_adapts_default = Int(round(N / 2))
                spl.alg.n_adapts = n_adapts_default > 1000 ? 1000 : n_adapts_default
            else
                # Verify that n_adapts is less than the samples to draw.
                spl.alg.n_adapts < N || !ismissing(resume_from) ?
                    nothing :
                    throw(ArgumentError("n_adapt of $(spl.alg.n_adapts) is greater than total samples of $N."))
            end
        else
            spl.alg.n_adapts = 0
        end
    end

    # Convert to transformed space if we're using
    # non-Gibbs sampling.
    if spl.selector.tag == :default
        link!(spl.state.vi, spl)
        runmodel!(model, spl.state.vi, spl)
    end
end

"""
    HMCDA(n_adapts::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.1)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(200, 0.65, 0.3)
```

Arguments:

- `n_adapts::Int` : Numbers of samples to use for adaptation.
- `δ::Float64` : Target acceptance rate. 65% is often recommended.
- `λ::Float64` : Target leapfrop length.
- `init_ϵ::Float64=0.1` : Inital step size; 0 means automatically search by Turing.

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

- Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively
  setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning
  Research 15, no. 1 (2014): 1593-1623.
"""
mutable struct HMCDA{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_adapts    ::  Int       # number of samples with adaption for ϵ
    δ           ::  Float64   # target accept rate
    λ           ::  Float64   # target leapfrog length
    init_ϵ      ::  Float64
end
HMCDA(args...; kwargs...) = HMCDA{ADBackend()}(args...; kwargs...)
function HMCDA{AD}(n_adapts::Int, δ::Float64, λ::Float64, init_ϵ::Float64, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
    return HMCDA{AD, space, metricT}(n_adapts, δ, λ, init_ϵ)
end

function HMCDA{AD}(
    δ::Float64,
    λ::Float64;
    init_ϵ::Float64=0.1,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMCDA{AD}(0, δ, λ, init_ϵ,metricT, ())
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
    init_ϵ::Float64=0.1,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMCDA{AD}(n_adapts, δ, λ, init_ϵ, metricT, space)
end


"""
    NUTS(n_adapts::Int, δ::Float64; max_depth::Int=5, Δ_max::Float64=1000.0, init_ϵ::Float64=0.1)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(200, 0.6j_max)
```

Arguments:

- `n_adapts::Int` : The number of samples to use with adapatation.
- `δ::Float64` : Target acceptance rate.
- `max_depth::Float64` : Maximum doubling tree depth.
- `Δ_max::Float64` : Maximum divergence during doubling tree.
- `init_ϵ::Float64` : Inital step size; 0 means automatically search by Turing.

"""

mutable struct NUTS{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_adapts    ::  Int       # number of samples with adaption for ϵ
    δ           ::  Float64   # target accept rate
    max_depth   ::  Int
    Δ_max       ::  Float64
    init_ϵ      ::  Float64
end

NUTS(args...; kwargs...) = NUTS{ADBackend()}(args...; kwargs...)
function NUTS{AD}(
    n_adapts::Int,
    δ::Float64,
    max_depth::Int,
    Δ_max::Float64,
    init_ϵ::Float64,
    ::Type{metricT},
    space::Tuple
) where {AD, metricT}
    return NUTS{AD, space, metricT}(n_adapts, δ, max_depth, Δ_max, init_ϵ)
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
    max_depth::Int=5,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.1,
    metricT=AHMC.DenseEuclideanMetric
) where AD
    NUTS{AD}(n_adapts, δ, max_depth, Δ_max, init_ϵ, metricT, space)
end

function NUTS{AD}(
    δ::Float64;
    max_depth::Int=5,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.1,
    metricT=AHMC.DenseEuclideanMetric
) where AD
    NUTS{AD}(0, δ, max_depth, Δ_max, init_ϵ, metricT, ())
end

function NUTS{AD}() where AD
    NUTS{AD}(0, 0.65, 5, 1000.0, 0.1, AHMC.DenseEuclideanMetric, ())
end

for alg in (:HMC, :HMCDA, :NUTS)
    @eval getmetricT(::$alg{<:Any, <:Any, metricT}) where {metricT} = metricT
end

####
#### Sampler construction
####

# Sampler(alg::Hamiltonian) =  Sampler(alg, AHMCAdaptor())
function Sampler(
    alg::Hamiltonian{AD},
    model::Model,
    s::Selector=Selector()
) where AD
    info = Dict{Symbol, Any}()
    state_bad = BlankState(VarInfo(model))

    # Create an initial sampler, to get all the initialization out of the way.
    spl_bad = Sampler(alg, info, s, state_bad)

    # Create the actual state based on the alg type.
    state = HMCState(model, spl_bad)

    # Create a real sampler after getting all the types/running the init phase.
    return Sampler(alg, spl_bad.info, spl_bad.selector, state)
end

####
#### Transition / step functions for HMC samplers.
####

# Single step of a Hamiltonian.
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{T},
    N::Integer;
    kwargs...
) where T<:Hamiltonian
    # Get step size
    ϵ = T <: AdaptiveHamiltonian ?
        AHMC.getϵ(spl.state.adaptor) :
        spl.alg.ϵ

    spl.state.i += 1
    spl.state.eval_num = 0

    Turing.DEBUG && @debug "current ϵ: $ϵ"

    # Transform the space if we're using Gibbs.
    Turing.DEBUG && @debug "X-> R..."
    if spl.selector.tag != :default
        link!(spl.state.vi, spl)
        runmodel!(model, spl.state.vi, spl)
    end

    grad_func = gen_∂logπ∂θ(spl.state.vi, spl, model)
    lj_func = gen_logπ(spl.state.vi, spl, model)
    metric = gen_metric(length(spl.state.vi[spl]), spl)

    θ, lj = spl.state.vi[spl], spl.state.vi.logp

    θ_new, lj_new, is_accept, α = hmc_step(θ, lj_func, grad_func, ϵ, spl.alg, metric)

    Turing.DEBUG && @debug "decide whether to accept..."
    if is_accept
        spl.state.vi[spl] = θ_new
        setlogp!(spl.state.vi, lj_new)
    else
        spl.state.vi[spl] = θ
        setlogp!(spl.state.vi, lj)
    end

    if T <: AdaptiveHamiltonian
        if spl.state.i <= spl.alg.n_adapts
            AHMC.adapt!(spl.state.adaptor, Vector{Float64}(spl.state.vi[spl]), α.acceptance_rate)
        end
    end

    # Transform the space back if we're using Gibbs.
    Turing.DEBUG && @debug "R -> X..."
    spl.selector.tag != :default && invlink!(spl.state.vi, spl)

    return transition(spl, α)
end


# Efficient multiple step sampling for adaptive HMC.
function steps!(model,
    spl::Sampler{<:AdaptiveHamiltonian},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true
)
    ahmc_samples = AHMC.sample(
        rng,
        spl.info[:h],
        spl.info[:traj],
        Vector{Float64}(vi[spl]),
        spl.alg.n_iters,
        spl.info[:adaptor],
        spl.alg.n_adapts;
        verbose=verbose
    )
    for i = 1:length(samples)
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
    end
end

# Efficient multiple step sampling for static HMC.
function steps!(
    model,
    spl::Sampler{<:HMC},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true
)
    ahmc_samples = AHMC.sample(
        rng,
        spl.info[:h],
        spl.info[:traj],
        Vector{Float64}(vi[spl]),
        spl.alg.n_iters;
        verbose=verbose
    )
    for i = 1:length(samples)
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
    end
end

# Default multiple step sampling for all HMC samplers.
function steps!(
    model,
    spl::Sampler{<:Hamiltonian},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true
)
    # Init step
    time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val(true))
    samples[1].value = Sample(vi, spl).value    # we know we always accept the init step
    samples[1].value[:elapsed] = time_elapsed
    # Rest steps
    for i = 2:length(samples)
        time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val(false))
        if is_accept # accepted => store the new predcits
            samples[i].value = Sample(vi, spl).value
        else         # rejected => store the previous predcits
            samples[i] = samples[i - 1]
        end
        samples[i].value[:elapsed] = time_elapsed
    end
end


#####
##### HMC core functions
#####

"""
    gen_∂logπ∂θ(vi::VarInfo, spl::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, spl, model)`.
"""
function gen_∂logπ∂θ(vi::VarInfo, spl::Sampler, model)
    function ∂logπ∂θ(x)::Tuple{Float64, Vector{Float64}}
        x_old, lj_old = vi[spl], vi.logp
        lp, deriv = gradient_logp(x, vi, model, spl)
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lp, deriv
    end
    return ∂logπ∂θ
end

"""
    gen_logπ(vi::VarInfo, spl::Sampler, model)

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, spl, model)`.
"""
function gen_logπ(vi::VarInfo, spl::Sampler, model)
    function logπ(x)::Float64
        x_old, lj_old = vi[spl], vi.logp
        vi[spl] = x
        runmodel!(model, vi, spl)
        lj = vi.logp
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

gen_metric(dim::Int, spl::Sampler{<:Hamiltonian}) = AHMC.UnitEuclideanMetric(dim)
gen_metric(dim::Int, ::AHMC.UnitPreconditioner)   = AHMC.UnitEuclideanMetric(dim)
gen_metric(::Int, pc::AHMC.DiagPreconditioner)    = AHMC.DiagEuclideanMetric(AHMC.getM⁻¹(pc))
gen_metric(::Int, pc::AHMC.DensePreconditioner)   = AHMC.DenseEuclideanMetric(AHMC.getM⁻¹(pc))
gen_metric(dim::Int, spl::Sampler{<:AdaptiveHamiltonian}) = gen_metric(dim, spl.state.adaptor.pc)

gen_traj(alg::HMC, ϵ) = AHMC.StaticTrajectory(AHMC.Leapfrog(ϵ), alg.n_leapfrog)
gen_traj(alg::HMCDA, ϵ) = AHMC.HMCDA(AHMC.Leapfrog(ϵ), alg.λ)
gen_traj(alg::NUTS, ϵ) = AHMC.NUTS(AHMC.Leapfrog(ϵ), alg.max_depth, alg.Δ_max)

function hmc_step(
    θ,
    logπ,
    ∂logπ∂θ,
    ϵ,
    alg::T,
    metric
) where {T<:Union{HMC,HMCDA,NUTS}}
    # Make sure the code in AHMC is type stable
    θ = Vector{Float64}(θ)

    # Build Hamiltonian type and trajectory
    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    traj = gen_traj(alg, ϵ)
    
    h = AHMC.update(h, θ) # Ensure h.metric has the same dim as θ.

    # Sample momentum
    r = AHMC.rand(h.metric)

    # Build phase point
    z = AHMC.phasepoint(h, θ, r)

    # TODO: remove below when we can get is_accept from AHMC.transition
    H = AHMC.neg_energy(z)  # NOTE: this a waste of computation

    # Call AHMC to make one MCMC transition
    z_new, α = AHMC.transition(traj, h, z)

    # Compute new Hamiltonian energy
    H_new = AHMC.neg_energy(z_new)
    θ_new = z_new.θ

    # NOTE: as `transition` doesn't return `is_accept`,
    #       I use `H == H_new` to check if the sample is accepted.
    is_accept = H != H_new  # If the new Hamiltonian enerygy is different
                            # from the old one, the sample was accepted.
    alg isa NUTS && (is_accept = true)  # we always accept in NUTS

    # Compute updated log-joint probability
    lj_new = logπ(θ_new)

    return θ_new, lj_new, is_accept, α
end

####
#### Compiler interface, i.e. tilde operators.
####

function assume(spl::Sampler{<:Hamiltonian},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    Turing.DEBUG && @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "vn = $vn"
    Turing.DEBUG && @debug "r = $r" "typeof(r)=$(typeof(r))"
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::Sampler{<:Hamiltonian},
    dists::Vector{<:Distribution},
    vn::VarName,
    var::Any,
    vi::VarInfo
)
    @assert length(dists) == 1 "[observe] Turing only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> VarName(vn, "[$i]"), 1:n)

    rs = vi[vns]  # NOTE: inside Turing the Julia conversion should be sticked to

    # acclogp!(vi, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1]))))

    if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
        @assert size(var) == size(rs) "Turing.assume variable and random number dimension unmatched"
        var = rs
    elseif isa(dist, MultivariateDistribution)
        if isa(var, Vector)
            @assert length(var) == size(rs)[2] "Turing.assume variable and random number dimension unmatched"
            for i = 1:n
                var[i] = rs[:,i]
            end
        elseif isa(var, Matrix)
            @assert size(var) == size(rs) "Turing.assume variable and random number dimension unmatched"
            var = rs
        else
            error("[Turing] unsupported variable container")
        end
    end

    var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))
end

observe(spl::Sampler{<:Hamiltonian},
    d::Distribution,
    value::Any,
    vi::VarInfo) = observe(nothing, d, value, vi)

observe(spl::Sampler{<:Hamiltonian},
    ds::Vector{<:Distribution},
    value::Any,
    vi::VarInfo) = observe(nothing, ds, value, vi)


####
#### Default HMC stepsize and mass matrix adaptor
####

function AHMCAdaptor(alg::AdaptiveHamiltonian)
    p = AHMC.Preconditioner(getmetricT(alg))
    nda = AHMC.NesterovDualAveraging(alg.δ, alg.init_ϵ)
    if getmetricT(alg) == AHMC.UnitEuclideanMetric
        adaptor = AHMC.NaiveHMCAdaptor(p, nda)
    else
        adaptor = AHMC.StanHMCAdaptor(alg.n_adapts, p, nda)
    end
    return adaptor
end


AHMCAdaptor(alg::Hamiltonian) = nothing

##########################
# HMC State Constructors #
##########################

function HMCState(model::Model, spl::Sampler{<:StaticHamiltonian}; kwargs...)
    # Reuse the VarInfo.
    vi = spl.state.vi

    # Get the metric type.
    metricT = getmetricT(spl.alg)

    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)
    h = AHMC.Hamiltonian(
        metricT(length(vi[spl])),
        logπ, ∂logπ∂θ)
    traj = gen_traj(spl.alg, spl.alg.ϵ)

    return StaticHMCState(vi, 0, 0, traj, h)
end

function HMCState(model::Model,
        spl::Sampler{<:AdaptiveHamiltonian};
        adaptor=AHMCAdaptor(spl.alg),
        kwargs...
)
    # Reuse the VarInfo.
    vi = spl.state.vi

    # Link everything if needed.
    link!(vi, spl)

    # Get the initial log pdf and gradient functions.
    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)

    # Get the metric type.
    metricT = getmetricT(spl.alg)

    # Create a Hamiltonian.
    θ_init = Vector{Float64}(spl.state.vi[spl])
    metric = metricT(length(θ_init))
    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)

    # Find a step size.
    init_ϵ = spl.alg.init_ϵ

    # Find good eps if not provided one
    if init_ϵ == 0.0
        init_ϵ = AHMC.find_good_eps(h, θ_init)
        @info "Found initial step size" init_ϵ
    end

    # Generate a trajectory.
    traj = gen_traj(spl.alg, init_ϵ)

    # Unlink everything.
    invlink!(vi, spl)

    return DynamicHMCState(vi, 0, 0, traj, h, adaptor)
end
