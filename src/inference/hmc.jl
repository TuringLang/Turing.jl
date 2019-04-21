"""
    HMC(n_iters::Int, epsilon::Float64, tau::Int)

Hamiltonian Monte Carlo sampler.

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `epsilon::Float64` : The leapfrog step size to use.
- `tau::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x[1] ~ Normal(m, sqrt(s))
    x[2] ~ Normal(m, sqrt(s))
    return s, m
end

sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))
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
mutable struct HMC{AD, T} <: StaticHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    epsilon   ::  Float64   # leapfrog step size
    tau       ::  Int       # leapfrog step number
    space     ::  Set{T}    # sampling space, emtpy means all
    metricT
end
HMC(args...) = HMC{ADBackend()}(args...)
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int; metricT=AHMC.UnitEuclideanMetric) where AD
    return HMC{AD, Any}(n_iters, epsilon, tau, Set(), metricT)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int, space...; metricT=AHMC.UnitEuclideanMetric) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMC{AD, eltype(_space)}(n_iters, epsilon, tau, _space, metricT)
end

function hmc_step(θ, lj, lj_func, grad_func, ϵ, alg::HMC, metric)
  θ_new, lj_new, is_accept, α = _hmc_step(θ, lj, lj_func, grad_func, ϵ, alg.tau, metric)
  return θ_new, lj_new, is_accept, α
end

# Below is a trick to remove the dependency of Stan by Requires.jl
# Please see https://github.com/TuringLang/Turing.jl/pull/459 for explanations
DEFAULT_ADAPT_CONF_TYPE = Nothing
STAN_DEFAULT_ADAPT_CONF = nothing

Sampler(alg::Hamiltonian, s::Selector) =  Sampler(alg, nothing, s)
Sampler(alg::Hamiltonian, adapt_conf::Nothing) = Sampler(alg, adapt_conf, Selector())
function Sampler(alg::Hamiltonian, adapt_conf::Nothing, s::Selector)
    return _sampler(alg::Hamiltonian, adapt_conf, s)
end
function _sampler(alg::Hamiltonian, adapt_conf, s::Selector=Selector())
    info = Dict{Symbol, Any}()

    # For state infomation
    info[:lf_num] = 0
    info[:eval_num] = 0

    # Adapt configuration
    info[:adapt_conf] = adapt_conf
    info[:i] = 0

    Sampler(alg, info, s)
end

function sample(model::Model, alg::Hamiltonian;
                save_state=false,                                   # flag for state saving
                resume_from=nothing,                                # chain to continue
                reuse_spl_n=0,                                      # flag for spl re-using
                adapt_conf=STAN_DEFAULT_ADAPT_CONF,                 # adapt configuration
                init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
                rng::AbstractRNG=GLOBAL_RNG,
                )

    # Create sampler
    spl = reuse_spl_n > 0 ?
          resume_from.info[:spl] :
          Sampler(alg, adapt_conf)
    if resume_from != nothing
        spl.selector = resume_from.info[:spl].selector
    end

    @assert isa(spl.alg, Hamiltonian) "[Turing] alg type mismatch; please use resume() to re-use spl"

    alg_str = isa(alg, HMC)   ? "HMC"   :
              isa(alg, HMCDA) ? "HMCDA" :
              isa(alg, SGHMC) ? "SGHMC" :
              isa(alg, SGLD)  ? "SGLD"  :
              isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

    # TODO: figure out what does this line do
    n = reuse_spl_n > 0 ?
        reuse_spl_n :
        alg.n_iters

    # Init samples
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Create VarInfo
    vi = if resume_from == nothing
        vi_ = VarInfo()
        model(vi_, SampleFromUniform())
        spl.info[:eval_num] += 1
        vi_
    else
        deepcopy(resume_from.info[:vi])
    end

    if init_theta != nothing
        println("Using init_theta=$init_theta")
        init_theta_flat = foldl(vcat, init_theta)
        theta_mask = map(x -> !ismissing(x), init_theta_flat)
        theta = vi[spl]
        theta[theta_mask] .= init_theta_flat[theta_mask]
        vi[spl] = theta
    end

    # Convert to transformed sapce
    if spl.selector.tag == :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    # Init h, prop and adaptor
    step(model, spl, vi, Val(true))

    # Sampling using AHMC
    if spl.alg isa AdaptiveHamiltonian
        ahmc_samples = AHMC.sample(rng, spl.info[:h], spl.info[:prop], Vector{Float64}(vi[spl]), spl.alg.n_iters, spl.info[:adaptor], spl.alg.n_adapts)
        for i = 1:n
            vi[spl] = ahmc_samples[i]
            samples[i].value = Sample(vi, spl).value
        end
    elseif spl.alg isa HMC
        ahmc_samples = AHMC.sample(rng, spl.info[:h], spl.info[:prop], Vector{Float64}(vi[spl]), spl.alg.n_iters)
        for i = 1:n
            vi[spl] = ahmc_samples[i]
            samples[i].value = Sample(vi, spl).value
        end
    else
        for i = 1:n
            time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val(i == 1))
            if is_accept # accepted => store the new predcits
                samples[i].value = Sample(vi, spl).value
            else         # rejected => store the previous predcits
                samples[i] = samples[i - 1]
            end
            samples[i].value[:elapsed] = time_elapsed
            if haskey(spl.info, :adaptor)
                samples[i].value[:lf_eps] = AHMC.getϵ(spl.info[:adaptor])
            end
        end
    end

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
    end

    c = Chain(0.0, samples)       # wrap the result by Chain

    if save_state               # save state
        # Convert vi back to X if vi is required to be saved
        spl.selector.tag == :default && invlink!(vi, spl)
        c = save(c, spl, model, vi, samples)
    end

    return c
end

function step(model, spl::Sampler{<:StaticHamiltonian}, vi::VarInfo, is_first::Val{true})
    ∂logπ∂θ = gen_grad_func(vi, spl, model)
    logπ = gen_lj_func(vi, spl, model)

    metric = spl.alg.metricT(length(vi[spl]))
    spl.info[:h] = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    ϵ = spl.alg.epsilon
    spl.info[:prop] = AHMC.StaticTrajectory(AHMC.Leapfrog(ϵ), spl.alg.tau)
    return vi, true
end

function step(model, spl::Sampler{<:AdaptiveHamiltonian}, vi::VarInfo, is_first::Val{true})
    spl.selector.tag != :default && link!(vi, spl)

    ∂logπ∂θ = gen_grad_func(vi, spl, model)
    logπ = gen_lj_func(vi, spl, model)

    θ_init = Vector{Float64}(vi[spl])
    metric = spl.alg.metricT(length(θ_init))

    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    init_ϵ = spl.alg.init_ϵ
    if init_ϵ == 0.0
        # @info h θ_init
        init_ϵ = AHMC.find_good_eps(h, θ_init)
        @info "Found initial step size" init_ϵ
    end
    if spl.alg isa NUTS
        prop = AHMC.NUTS(AHMC.Leapfrog(init_ϵ), spl.alg.max_depth, spl.alg.Δ_max)
    elseif spl.alg isa HMCDA
        prop = AHMC.HMCDA(AHMC.Leapfrog(init_ϵ), spl.alg.λ)
    else
        @error "Unsupported adaptive algorithm" spl.alg
    end
    adaptor = AHMC.StanNUTSAdaptor(spl.alg.n_adapts, AHMC.PreConditioner(metric),
                                          AHMC.NesterovDualAveraging(spl.alg.δ, init_ϵ))

    spl.info[:h] = h
    spl.info[:prop] = prop
    spl.info[:adaptor] = adaptor

    spl.selector.tag != :default && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:Hamiltonian}, vi::VarInfo, is_first::Val{false})
    # Get step size
    if :adaptor in keys(spl.info)
        ϵ = AHMC.getϵ(spl.info[:adaptor])
    else
        ϵ = spl.alg.epsilon
    end
    spl.info[:i] += 1
    Turing.DEBUG && @debug "current ϵ: $ϵ"

    # Reset current counters
    spl.info[:lf_num] = 0
    spl.info[:eval_num] = 0

    Turing.DEBUG && @debug "X-> R..."
    if spl.selector.tag != :default
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    grad_func = gen_grad_func(vi, spl, model)
    lj_func = gen_lj_func(vi, spl, model)
    if :adaptor in keys(spl.info)
        metric = gen_metric(vi, spl, spl.info[:adaptor].pc)
    else
        metric = gen_metric(vi, spl)
    end

    θ, lj = vi[spl], vi.logp

    θ_new, lj_new, is_accept, α = hmc_step(θ, lj, lj_func, grad_func, ϵ, spl.alg, metric)

    Turing.DEBUG && @debug "decide whether to accept..."
    if is_accept
        vi[spl] = θ_new
        setlogp!(vi, lj_new)
    else
        vi[spl] = θ
        setlogp!(vi, lj)
    end

    if PROGRESS[] && spl.selector.tag == :default
        metric = gen_metric(vi, spl, spl.info[:adaptor].pc)
        haskey(spl.info, :progress) && ProgressMeter.update!(
            spl.info[:progress],
            spl.info[:progress].counter;
            showvalues = [(:ϵ, ϵ), (:α, α), (:metric, metric)],
        )
    end

    if spl.alg isa AdaptiveHamiltonian
        if spl.info[:i] <= spl.alg.n_adapts
            # AHMC.adapt!(spl.info[:adaptor], Vector{Float64}(vi[spl]), α)
        end
    end

    Turing.DEBUG && @debug "R -> X..."
    spl.selector.tag != :default && invlink!(vi, spl)

    return vi, is_accept
end

###

function assume(spl::Sampler{<:Hamiltonian}, dist::Distribution, vn::VarName, vi::VarInfo)
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

function assume(spl::Sampler{<:Hamiltonian}, dists::Vector{<:Distribution}, vn::VarName, var::Any, vi::VarInfo)
    @assert length(dists) == 1 "[observe] Turing only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

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

observe(spl::Sampler{<:Hamiltonian}, d::Distribution, value::Any, vi::VarInfo) =
    observe(nothing, d, value, vi)

observe(spl::Sampler{<:Hamiltonian}, ds::Vector{<:Distribution}, value::Any, vi::VarInfo) =
    observe(nothing, ds, value, vi)
