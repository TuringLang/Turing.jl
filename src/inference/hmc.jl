###
### Hamiltonian Monte Carlo samplers.
###

"""
    HMC(n_iters::Int, ϵ::Float64, n_leapfrog::Int)

Hamiltonian Monte Carlo sampler.

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `ϵ::Float64` : The leapfrog step size to use.
- `n_leapfrog::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
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
    n_iters     ::  Int       # number of samples
    ϵ           ::  Float64   # leapfrog step size
    n_leapfrog  ::  Int       # leapfrog step number
end
HMC(args...) = HMC{ADBackend()}(args...)
function HMC{AD}(n_iters::Int, ϵ::Float64, n_leapfrog::Int, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
    return HMC{AD, space, metricT}(n_iters, ϵ, n_leapfrog)
end
function HMC{AD}(
    n_iters::Int,
    ϵ::Float64,
    n_leapfrog::Int,
    ::Tuple{};
    kwargs...
) where AD
    return HMC{AD}(n_iters, ϵ, n_leapfrog; kwargs...)
end
function HMC{AD}(
    n_iters::Int,
    ϵ::Float64,
    n_leapfrog::Int,
    space::Symbol...;
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMC{AD}(n_iters, ϵ, n_leapfrog, metricT, space)
end

"""
    HMCDA(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.1)

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
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
    n_iters     ::  Int       # number of samples
    n_adapts    ::  Int       # number of samples with adaption for ϵ
    δ           ::  Float64   # target accept rate
    λ           ::  Float64   # target leapfrog length
    init_ϵ      ::  Float64
end
HMCDA(args...; kwargs...) = HMCDA{ADBackend()}(args...; kwargs...)
function HMCDA{AD}(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64, init_ϵ::Float64, ::Type{metricT}, space::Tuple) where {AD, metricT <: AHMC.AbstractMetric}
    return HMCDA{AD, space, metricT}(n_iters, n_adapts, δ, λ, init_ϵ)
end

function HMCDA{AD}(
    n_iters::Int,
    δ::Float64,
    λ::Float64;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    n_adapts_default = Int(round(n_iters / 2))
    n_adapts = n_adapts_default > 1000 ? 1000 : n_adapts_default
    return HMCDA{AD}(n_iters, n_adapts, δ, λ, init_ϵ,metricT, ())
end

function HMCDA{AD}(
    n_iters::Int,
    n_adapts::Int,
    δ::Float64,
    λ::Float64,
    ::Tuple{};
    kwargs...
) where AD
    return HMCDA{AD}(n_iters, n_adapts, δ, λ; kwargs...)
end

function HMCDA{AD}(
    n_iters::Int,
    n_adapts::Int,
    δ::Float64,
    λ::Float64,
    space::Symbol...;
    init_ϵ::Float64=0.0,
    metricT=AHMC.UnitEuclideanMetric
) where AD
    return HMCDA{AD}(n_iters, n_adapts, δ, λ, init_ϵ, metricT, space)
end

"""
    NUTS(n_iters::Int, n_adapts::Int, δ::Float64; max_depth::Int=5, Δ_max::Float64=1000.0, init_ϵ::Float64=0.1)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.6j_max)
```

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `n_adapts::Int` : The number of samples to use with adapatation.
- `δ::Float64` : Target acceptance rate.
- `max_depth::Float64` : Maximum doubling tree depth.
- `Δ_max::Float64` : Maximum divergence during doubling tree.
- `init_ϵ::Float64` : Inital step size; 0 means automatically search by Turing.

"""
mutable struct NUTS{AD, space, metricT <: AHMC.AbstractMetric} <: AdaptiveHamiltonian{AD}
    n_iters     ::  Int       # number of samples
    n_adapts    ::  Int       # number of samples with adaption for ϵ
    δ           ::  Float64   # target accept rate
    max_depth   ::  Int
    Δ_max       ::  Float64
    init_ϵ      ::  Float64
end

NUTS(args...; kwargs...) = NUTS{ADBackend()}(args...; kwargs...)
function NUTS{AD}(
    n_iters::Int, 
    n_adapts::Int, 
    δ::Float64, 
    max_depth::Int, 
    Δ_max::Float64, 
    init_ϵ::Float64, 
    ::Type{metricT}, 
    space::Tuple
) where {AD, metricT}
    return NUTS{AD, space, metricT}(n_iters, n_adapts, δ, max_depth, Δ_max, init_ϵ)
end

function NUTS{AD}(
    n_iters::Int,
    n_adapts::Int,
    δ::Float64,
    ::Tuple{};
    kwargs...
) where AD
    NUTS{AD}(n_iters, n_adapts, δ; kwargs...)
end

function NUTS{AD}(
    n_iters::Int,
    n_adapts::Int,
    δ::Float64,
    space::Symbol...;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric
) where AD
    NUTS{AD}(n_iters, n_adapts, δ, max_depth, Δ_max, init_ϵ, metricT, space)
end

function NUTS{AD}(
    n_iters::Int,
    δ::Float64;
    max_depth::Int=10,
    Δ_max::Float64=1000.0,
    init_ϵ::Float64=0.0,
    metricT=AHMC.DiagEuclideanMetric
) where AD
    n_adapts_default = Int(round(n_iters / 2))
    NUTS{AD}(n_iters, n_adapts_default > 1000 ?
        1000 : n_adapts_default, δ, max_depth, Δ_max, init_ϵ, metricT, ())
end

for alg in (:HMC, :HMCDA, :NUTS)
    @eval getmetricT(::$alg{<:Any, <:Any, metricT}) where {metricT} = metricT
end

####
#### Sampler construction
####

# Sampler(alg::Hamiltonian) =  Sampler(alg, AHMCAdaptor())
function Sampler(alg::Hamiltonian, s::Selector=Selector())
    info = Dict{Symbol, Any}()

    info[:eval_num] = 0
    info[:i] = 0

    Sampler(alg, info, s)
end


function sample(
    model::Model,
    alg::Hamiltonian;
    save_state=false,                                   # flag for state saving
    resume_from=nothing,                                # chain to continue
    reuse_spl_n=0,                                      # flag for spl re-using
    adaptor=AHMCAdaptor(alg),
    init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
    rng::AbstractRNG=GLOBAL_RNG,
    discard_adapt::Bool=true,
    verbose::Bool=true,
    progress::Bool=false,
    kwargs...
)
    # Create sampler
    spl = reuse_spl_n > 0 ? resume_from.info[:spl] : Sampler(alg)
    @assert isa(spl.alg, Hamiltonian) "[Turing] alg type mismatch; please use resume() to re-use spl"

    # Resume selector
    resume_from != nothing && (spl.selector = resume_from.info[:spl].selector)

    # TODO: figure out what does this line do
    n = reuse_spl_n > 0 ? reuse_spl_n : alg.n_iters

    # Init samples
    samples = Vector{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Create VarInfo
    vi = if resume_from == nothing
        VarInfo(model)
    else
        deepcopy(resume_from.info[:vi])
    end

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
        vi[spl] = theta
    end

    # Convert to transformed space
    if spl.selector.tag == :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    # Init h, prop and adaptor
    step(model, spl, vi, Val(true); rng=rng, adaptor=adaptor)

    # Sampling using AHMC and store samples in `samples`
    steps!(model, spl, vi, samples; rng=rng, verbose=verbose, progress=progress)

    # Concatenate samples
    if resume_from != nothing
        pushfirst!(samples, resume_from.info[:samples]...)
    end

    # Wrap the result by Chain
    c = typeof(alg) <: AdaptiveHamiltonian && discard_adapt ?
        Chain(0.0, samples[(alg.n_adapts+1):end]) :
        Chain(0.0, samples)

    # Save state
    if save_state
        # Convert vi back to X if vi is required to be saved.
        spl.selector.tag == :default && invlink!(vi, spl)
        c = save(c, spl, model, vi, samples)
    end
    
    return c
end


####
#### Transition / step functions for HMC samplers.
####

# Init for StaticHamiltonian
function step(
    model,
    spl::Sampler{<:StaticHamiltonian},
    vi::VarInfo,
    is_first::Val{true};
    kwargs...
)
    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)
    spl.info[:h] = AHMC.Hamiltonian(getmetricT(spl.alg)(length(vi[spl])), logπ, ∂logπ∂θ)
    spl.info[:traj] = gen_traj(spl.alg, spl.alg.ϵ)
    return vi, true
end

# Init for AdaptiveHamiltonian
function step(
    model,
    spl::Sampler{<:AdaptiveHamiltonian},
    vi::VarInfo,
    is_first::Val{true};
    adaptor=AHMCAdaptor(spl.alg),
    rng::AbstractRNG=GLOBAL_RNG,
    kwargs...
)
    spl.selector.tag != :default && link!(vi, spl)

    ∂logπ∂θ = gen_∂logπ∂θ(vi, spl, model)
    logπ = gen_logπ(vi, spl, model)

    θ_init = Vector{Float64}(vi[spl])
    metric = getmetricT(spl.alg)(length(θ_init))
    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    init_ϵ = spl.alg.init_ϵ

    # Find good eps if not provided one
    if init_ϵ == 0.0
        init_ϵ = AHMC.find_good_eps(rng, h, θ_init)
        @info "Found initial step size" init_ϵ
    end
    if AHMC.getϵ(adaptor) == 0.0
        adaptor = AHMCAdaptor(spl.alg; init_ϵ=init_ϵ)
    end

    spl.info[:h] = h
    spl.info[:traj] = gen_traj(spl.alg, init_ϵ)
    spl.info[:adaptor] = adaptor

    spl.selector.tag != :default && invlink!(vi, spl)
    return vi, true
end

# Single step for Gibbs compatible HMC sampling.
function step(
    model,
    spl::Sampler{<:Hamiltonian},
    vi::VarInfo,
    is_first::Val{false}
)
    # Get step size
    ϵ = :adaptor in keys(spl.info) ? AHMC.getϵ(spl.info[:adaptor]) : spl.alg.ϵ

    spl.info[:i] += 1
    spl.info[:eval_num] = 0

    Turing.DEBUG && @debug "current ϵ: $ϵ"

    Turing.DEBUG && @debug "X-> R..."
    if spl.selector.tag != :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    grad_func = gen_∂logπ∂θ(vi, spl, model)
    lj_func = gen_logπ(vi, spl, model)
    metric = gen_metric(length(vi[spl]), spl)

    θ, lj = vi[spl], vi.logp

    θ_new, lj_new, is_accept, α = hmc_step(θ, lj_func, grad_func, ϵ, spl.alg, metric)

    Turing.DEBUG && @debug "decide whether to accept..."
    if is_accept
        vi[spl] = θ_new
        setlogp!(vi, lj_new)
    else
        vi[spl] = θ
        setlogp!(vi, lj)
    end

    if PROGRESS[] && spl.selector.tag == :default
        haskey(spl.info, :progress) && ProgressMeter.update!(
            spl.info[:progress],
            spl.info[:progress].counter;
            showvalues = [(:ϵ, ϵ), (:α, α), (:metric, metric)],
        )
    end

    if spl.alg isa AdaptiveHamiltonian
        if spl.info[:i] <= spl.alg.n_adapts
            AHMC.adapt!(spl.info[:adaptor], Vector{Float64}(vi[spl]), α)
        end
    end

    Turing.DEBUG && @debug "R -> X..."
    spl.selector.tag != :default && invlink!(vi, spl)

    return vi, is_accept
end


# Efficient multiple step sampling for adaptive HMC.
function steps!(
    model,
    spl::Sampler{<:AdaptiveHamiltonian},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true,
    progress::Bool=false
)
    ahmc_samples, stats = AHMC.sample(
        rng,
        spl.info[:h],
        spl.info[:traj],
        Vector{Float64}(vi[spl]),
        spl.alg.n_iters,
        spl.info[:adaptor],
        spl.alg.n_adapts;
        verbose=verbose,
        progress=progress
    )
    for i = 1:length(samples)
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
        foreach(name -> samples[i].value[name] = stats[i][name], typeof(stats[i]).names)
    end
end

# Efficient multiple step sampling for static HMC.
function steps!(
    model,
    spl::Sampler{<:HMC},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true,
    progress::Bool=false
)
    ahmc_samples, stats = AHMC.sample(
        rng,
        spl.info[:h],
        spl.info[:traj],
        Vector{Float64}(vi[spl]),
        spl.alg.n_iters;
        verbose=verbose,
        progress=progress
    )
    
    for i = 1:length(samples)
        vi[spl] = ahmc_samples[i]
        samples[i].value = Sample(vi, spl).value
        foreach(name -> samples[i].value[name] = stats[i][name], typeof(stats[i]).names)
    end
end

# Default multiple step sampling for all HMC samplers.
function steps!(
    model,
    spl::Sampler{<:Hamiltonian},
    vi,
    samples;
    rng::AbstractRNG=GLOBAL_RNG,
    verbose::Bool=true,
    progress::Bool=false
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
    function ∂logπ∂θ(x)
        return gradient_logp(x, vi, model, spl)
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
gen_metric(dim::Int, spl::Sampler{<:AdaptiveHamiltonian}) = gen_metric(dim, spl.info[:adaptor].pc)

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

    # Call AHMC to make one MCMC transition
    z_new, stat = AHMC.transition(traj, h, z)

    return z_new.θ, stat.log_density, stat.is_accept, stat.acceptance_rate
end

####
#### Compiler interface, i.e. tilde operators.
####
using Tracker
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
    r, logpdf_with_trans(dist, r, istrans(vi, vn))
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

function AHMCAdaptor(alg::AdaptiveHamiltonian; init_ϵ=alg.init_ϵ)
    p = AHMC.Preconditioner(getmetricT(alg))
    nda = AHMC.NesterovDualAveraging(alg.δ, init_ϵ)
    if getmetricT(alg) == AHMC.UnitEuclideanMetric
        adaptor = AHMC.NaiveHMCAdaptor(p, nda)
    else
        adaptor = AHMC.StanHMCAdaptor(alg.n_adapts, p, nda)
    end
    return adaptor
end


AHMCAdaptor(alg::Hamiltonian) = nothing
