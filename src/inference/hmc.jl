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
    gid       ::  Int       # group ID
end
HMC(args...) = HMC{ADBackend()}(args...)
function HMC{AD}(epsilon::Float64, tau::Int, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMC{AD, eltype(_space)}(1, epsilon, tau, _space, 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int) where AD
    return HMC{AD, Any}(n_iters, epsilon, tau, Set(), 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMC{AD, eltype(_space)}(n_iters, epsilon, tau, _space, 0)
end
function HMC{AD1}(alg::HMC{AD2, T}, new_gid::Int) where {AD1, AD2, T}
    return HMC{AD1, T}(alg.n_iters, alg.epsilon, alg.tau, alg.space, new_gid)
end
function HMC{AD, T}(alg::HMC, new_gid::Int) where {AD, T}
    return HMC{AD, T}(alg.n_iters, alg.epsilon, alg.tau, alg.space, new_gid)
end

function hmc_step(θ, lj, lj_func, grad_func, H_func, ϵ, alg::HMC, momentum_sampler::Function;
                  rev_func=nothing, log_func=nothing)
    θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(
        θ, lj, lj_func, grad_func, H_func, alg.tau, ϵ, momentum_sampler; rev_func=rev_func, log_func=log_func)
    return θ_new, lj_new, HMCStats(α, is_accept, ϵ, τ_valid)
end

# Below is a trick to remove the dependency of Stan by Requires.jl
# Please see https://github.com/TuringLang/Turing.jl/pull/459 for explanations
DEFAULT_ADAPT_CONF_TYPE = Nothing
STAN_DEFAULT_ADAPT_CONF = nothing

Sampler(alg::Hamiltonian) =  Sampler(alg, nothing)
function Sampler(alg::Hamiltonian, adapt_conf::Nothing)
    return _sampler(alg::Hamiltonian, adapt_conf)
end
function _sampler(alg::Hamiltonian, adapt_conf)
    info=Dict{Symbol, Any}()

    # Adapt configuration
    info[:adapt_conf] = adapt_conf

    Sampler(alg, info)
end

function sample(model::Model, alg::Hamiltonian;
                                save_state=false,                   # flag for state saving
                                resume_from=nothing,                # chain to continue
                                reuse_spl_n=0,                      # flag for spl re-using
                                adapt_conf=STAN_DEFAULT_ADAPT_CONF, # adapt configuration
                )
    spl = reuse_spl_n > 0 ?
          resume_from.info[:spl] :
          Sampler(alg, adapt_conf)

    @assert isa(spl.alg, Hamiltonian) "[Turing] alg type mismatch; please use resume() to re-use spl"

    alg_str = isa(alg, HMC)   ? "HMC"   :
              isa(alg, HMCDA) ? "HMCDA" :
              isa(alg, SGHMC) ? "SGHMC" :
              isa(alg, SGLD)  ? "SGLD"  :
              isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

    # Initialization
    time_total = zero(Float64)
    n = reuse_spl_n > 0 ?
        reuse_spl_n :
        alg.n_iters
    samples = Array{Sample}(undef, n)
    weight = 1 / n
    for i = 1:n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    vi = if resume_from == nothing
        vi_ = VarInfo()
        model(vi_, HamiltonianRobustInit())
        vi_
    else
        deepcopy(resume_from.info[:vi])
    end

    if spl.alg.gid == 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    # HMC steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    local stats
    for i = 1:n
        @debug "$alg_str stepping..."

        time_elapsed = @elapsed vi, stats = step(model, spl, vi, Val(i == 1))
        time_total += time_elapsed

        if stats.is_accept  # accepted => store the new predcits
            samples[i].value = Sample(vi, stats; elapsed=time_elapsed).value
        else                # rejected => store the previous predcits
            samples[i] = samples[i - 1]
        end

        push!(accept_his, stats.is_accept)
        PROGRESS[] && ProgressMeter.next!(spl.info[:progress])
    end

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    println("[$alg_str] Finished with")
    println("  Running time        = $time_total;")
    # TODO: @Cameron we can simplify below when we have section for MCMCChain
    fns = fieldnames(typeof(stats))
    :_accept_ratio in fns && println("  alpha / sample      = $(mean(c[:_accept_ratio]));")
    :_is_accept    in fns && println("  Accept rate         = $(mean(c[:_is_accept]));")
    :_n_lf_steps   in fns && println("  #lf / sample        = $(mean(c[:_n_lf_steps]));")
    if haskey(spl.info, :wum)
      std_str = string(spl.info[:wum].pc)
      std_str = length(std_str) >= 32 ? std_str[1:30]*"..." : std_str   # only show part of pre-cond
      println("  pre-cond. metric    = $(std_str).")
    end


    if save_state               # save state
        # Convert vi back to X if vi is required to be saved
        if spl.alg.gid == 0 invlink!(vi, spl) end
        save!(c, spl, model, vi)
    end
    return c
end

function step(model, spl::Sampler{<:StaticHamiltonian}, vi::VarInfo, is_first::Val{true})
    spl.info[:wum] = NaiveCompAdapter(UnitPreConditioner(), FixedStepSize(spl.alg.epsilon))
    return vi, HMCStats(1.0, true, spl.alg.epsilon, 0)
end

function step(model, spl::Sampler{<:AdaptiveHamiltonian}, vi::VarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)
    ϵ = find_good_eps(model, spl, vi) # heuristically find good initial step size
    dim = length(vi[spl])
    spl.info[:wum] = ThreePhaseAdapter(spl, ϵ, dim)
    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, HMCStats(1.0, true, ϵ, 0)
end

function step(model, spl::Sampler{<:Hamiltonian}, vi::VarInfo, is_first::Val{false})
    @debug "current ϵ: $ϵ"
    ϵ = getss(spl.info[:wum])

    @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    grad_func = gen_grad_func(vi, spl, model)
    lj_func = gen_lj_func(vi, spl, model)
    rev_func = gen_rev_func(vi, spl)
    momentum_sampler = gen_momentum_sampler(vi, spl, spl.info[:wum].pc)
    H_func = gen_H_func(spl.info[:wum].pc)

    θ, lj = vi[spl], vi.logp

    θ_new, lj_new, stats = hmc_step(θ, lj, lj_func, grad_func, H_func, ϵ, spl.alg, momentum_sampler;
                                    rev_func=rev_func)
    α = stats.accept_ratio

    @debug "decide whether to accept..."
    if stats.is_accept
        vi[spl] = θ_new
        setlogp!(vi, lj_new)
    else
        vi[spl] = θ
        setlogp!(vi, lj)
    end

    if PROGRESS[] && spl.alg.gid == 0
        std_str = string(spl.info[:wum].pc)
        std_str = length(std_str) >= 32 ? std_str[1:30]*"..." : std_str
        haskey(spl.info, :progress) && ProgressMeter.update!(
            spl.info[:progress], spl.info[:progress].counter;
            showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, std_str)],
        )
    end

    if spl.alg isa AdaptiveHamiltonian
        # vi2 = deepcopy(vi)
        # invlink!(vi2, spl)
        # adapt!(spl.info[:wum], α, vi2[spl])
        adapt!(spl.info[:wum], α, vi[spl])
    end

    @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    return vi, stats
end

function assume(spl::Sampler{<:Hamiltonian}, dist::Distribution, vn::VarName, vi::VarInfo)
    @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    @debug "dist = $dist"
    @debug "vn = $vn"
    @debug "r = $r" "typeof(r)=$(typeof(r))"
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
