"""
    HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64)

Hamiltonian Monte Carlo sampler wiht Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
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

sample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))
```
"""
mutable struct HMCDA{T} <: Hamiltonian
  n_iters   ::  Int       # number of samples
  n_adapt   ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  lambda    ::  Float64   # target leapfrog length
  space     ::  Set{T}    # sampling space, emtpy means all
  gid       ::  Int       # group ID
end
function HMCDA(n_adapt::Int, delta::Float64, lambda::Float64, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMCDA(1, n_adapt, delta, lambda, _space, 0)
end
function HMCDA(n_iters::Int, delta::Float64, lambda::Float64)
    n_adapt_default = Int(round(n_iters / 2))
    n_adapt = n_adapt_default > 1000 ? 1000 : n_adapt_default
    return HMCDA(n_iters, n_adapt, delta, lambda, Set(), 0)
end
function HMCDA(alg::HMCDA, new_gid::Int)
    return HMCDA(alg.n_iters, alg.n_adapt, alg.delta, alg.lambda, alg.space, new_gid)
end
HMCDA{T}(alg::HMCDA, new_gid::Int) where {T} = HMCDA(alg, new_gid)
function HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64)
    return HMCDA(n_iters, n_adapt, delta, lambda, Set(), 0)
end
function HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return HMCDA(n_iters, n_adapt, delta, lambda, _space, 0)
end

function step(model, spl::Sampler{<:HMCDA}, vi::VarInfo, is_first::Bool)
    if is_first
        if ~haskey(spl.info, :wum)
            spl.alg.gid != 0 && link!(vi, spl)

            init_warm_up_params(vi, spl)

            θ = vi[spl]
            ϵ = spl.alg.delta > 0 ?
                find_good_eps(model, vi, spl) :       # heuristically find optimal ϵ
                spl.info[:pre_set_ϵ]
            vi[spl] = θ

            spl.alg.gid != 0 && invlink!(vi, spl)
            push!(spl.info[:wum][:ϵ], ϵ)
            update_da_μ(spl.info[:wum], ϵ)
        end

        push!(spl.info[:accept_his], true)
    else
        # Set parameters
        λ = spl.alg.lambda
        ϵ = spl.info[:wum][:ϵ][end]
        @debug "current ϵ: $ϵ"

        spl.info[:lf_num] = 0   # reset current lf num counter

        @debug "X-> R..."
        if spl.alg.gid != 0
            link!(vi, spl)
            runmodel!(model, vi, spl)
        end

        grad_func = gen_grad_func(vi, spl, model)
        lj_func = gen_lj_func(vi, spl, model)
        rev_func = gen_rev_func(vi, spl)
        log_func = gen_log_func(spl)

        θ, lj, stds = vi[spl], vi.logp, spl.info[:wum][:stds]

        θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(
            θ, lj, lj_func, grad_func, ϵ, λ, stds; rev_func=rev_func, log_func=log_func)

        if PROGRESS[] && spl.alg.gid == 0
            stds_str = string(spl.info[:wum][:stds])
            stds_str = length(stds_str) >= 32 ? stds_str[1:30]*"..." : stds_str
            haskey(spl.info, :progress) && ProgressMeter.update!(
                spl.info[:progress],
                spl.info[:progress].counter;
                showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, stds_str)],
            )
        end

        @debug "decide whether to accept..."
        if is_accept
            push!(spl.info[:accept_his], true)
            vi[spl] = θ_new
            setlogp!(vi, lj_new)
        else
            push!(spl.info[:accept_his], false)
            vi[spl] = θ
            setlogp!(vi, lj)
        end

        # QUES: why do we need the 2nd condition here (Kai)
        if spl.alg.delta > 0
            # TODO: figure out whether or not the condition below is needed
            # if spl.alg.delta > 0 && τ_valid > 0    # only do adaption for HMCDA
            adapt!(spl.info[:wum], α, vi[spl], adapt_ϵ = true)
        end

        @debug "R -> X..."
        spl.alg.gid != 0 && invlink!(vi, spl)
    end
    return vi
end

function _hmc_step(
    θ::AbstractVector{<:Real},
    lj::Real,
    lj_func,
    grad_func,
    ϵ::Real,
    λ::Real,
    stds::AbstractVector{<:Real};
    rev_func=nothing,
    log_func=nothing,
)

    θ_dim = length(θ)

    @debug "sampling momentums..."
    p = _sample_momentum(θ_dim, stds)

    @debug "recording old values..."
    H = _find_H(θ, p, lj, stds)

    τ = max(1, round(Int, λ / ϵ))
    @debug "leapfrog for $τ steps with step size $ϵ"
    θ_new, p_new, τ_valid = _leapfrog(θ, p, τ, ϵ, grad_func; rev_func=rev_func, log_func=log_func)

    @debug "computing new H..."
    lj_new = lj_func(θ_new)
    H_new = (τ_valid == 0) ? Inf : _find_H(θ_new, p_new, lj_new, stds)

    @debug "deciding wether to accept and computing accept rate α..."
    is_accept, logα = mh_accept(H, H_new)

    if is_accept
        θ = θ_new
        lj = lj_new
    end

    return θ, lj, is_accept, τ_valid, exp(logα)
end

function _hmc_step(
    θ::AbstractVector{<:Real},
    lj::Real,
    lj_func,
    grad_func,
    τ::Int,
    ϵ::Real,
    stdsstds::AbstractVector{<:Real},
)
    return _hmc_step(θ, lj, lj_func, grad_func, ϵ, τ * ϵ, stds)
end
