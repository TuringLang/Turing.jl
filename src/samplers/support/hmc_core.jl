# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp

"""
    gen_grad_func(vi::VarInfo, sampler::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, sampler, model)`.
"""
function gen_grad_func(vi::VarInfo, sampler::Sampler, model)
    return θ::AbstractVector{<:Real}->gradient(θ, vi, model, sampler)
end

"""
    gen_lj_func(vi::VarInfo, sampler::Sampler, model)

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, sampler, model)`.
"""
function gen_lj_func(vi::VarInfo, sampler::Sampler, model)
    return function(θ::AbstractVector{<:Real})
        vi[sampler] = θ
        return runmodel!(model, vi, sampler).logp
    end
end

"""
  gen_rev_func(vi::VarInfo, sampler::Sampler)

Generate a function on `(θ, logp)` that sets the variables referenced by `sampler` to `θ`
and the current `vi.logp` to `logp`.
"""
function gen_rev_func(vi::VarInfo, sampler::Sampler)
    return function(θ::AbstractVector{<:Real}, logp::Real)
        vi[sampler] = θ
        setlogp!(vi, logp)
    end
end

"""
    gen_log_func(sampler::Sampler)

Generate a function that takes no argument and performs logging for the number of leapfrog
steps used in `sampler`.
"""
function gen_log_func(sampler::Sampler)
    return function()
        sampler.info[:lf_num] += 1
    end
end

_sample_momentum(d::Int, std::Vector) = randn(d) ./ std

function runmodel!(model::Function, vi::VarInfo, spl::Union{Nothing,Sampler})
    setlogp!(vi, zero(Real))
    if spl != nothing && :eval_num ∈ keys(spl.info)
        spl.info[:eval_num] += 1
    end
    Base.invokelatest(model, vi, spl)
    return vi
end

function leapfrog(
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    τ::Int,
    ϵ::Real,
    model::Function,
    vi::VarInfo,
    sampler::Sampler,
)
    lp_grad = gen_grad_func(vi, sampler, model)
    rev = gen_rev_func(vi, sampler)
    logger = gen_log_func(sampler)
    return _leapfrog(θ, p, τ, ϵ, lp_grad; rev_func=rev, log_func=logger)
end

function _leapfrog(
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    τ::Int,
    ϵ::Real,
    lp_grad_func::Function;
    rev_func=nothing,
    log_func=nothing,
)
    _, grad = lp_grad_func(θ)
    verifygrad(grad) || (return θ, p, 0)

    p, θ, τ_valid = deepcopy(p), deepcopy(θ), 0

    p .-= ϵ .* grad ./ 2
    for t in 1:τ

        log_func != nothing && log_func()

        θ .+= ϵ .* p
        logp, grad = lp_grad_func(θ)

        # If gradients explode, tidy up and return.
        if ~verifygrad(grad)
            θ .-= ϵ .* p
            rev_func != nothing && rev_func(θ, logp)
            break
        end

        p .-= ϵ .* grad
        τ_valid += 1
    end

    # Undo half a step in the momenta.
    p .+= ϵ .* grad ./ 2

    return θ, p, τ_valid
end

# Compute Hamiltonian
function _find_H(
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    logpdf_func_float::Function,
    std::AbstractVector{<:Real},
)
    return _find_H(θ, p, logpdf_func_float(θ), std)
end

function _find_H(
    θ::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    logp::Real,
    std::AbstractVector{<:Real},
)
    H = sum(abs2, p .* std) / 2 - logp
    return isnan(H) ? Inf : H
end

function _hmc_step(
    θ::AbstractVector{<:Real},
    lj::Real,
    lj_func,
    grad_func,
    τ::Int,
    ϵ::Real,
    std::AbstractVector{<:Real};
    rev_func=nothing,
    log_func=nothing,
)

    θ_dim = length(θ)

    @debug "sampling momentums..."
    p = _sample_momentum(θ_dim, std)

    @debug "recording old values..."
    H = _find_H(θ, p, lj, std)


    @debug "leapfrog for $τ steps with step size $ϵ"
    θ_new, p_new, τ_valid = _leapfrog(θ, p, τ, ϵ, grad_func; rev_func=rev_func, log_func=log_func)

    @debug "computing new H..."
    lj_new = lj_func(θ_new)
    H_new = (τ_valid == 0) ? Inf : _find_H(θ_new, p_new, lj_new, std)

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
    ϵ::Real,
    λ::Real,
    std::AbstractVector{<:Real};
    rev_func=nothing,
    log_func=nothing,
)
    τ = max(1, round(Int, λ / ϵ))
    return _hmc_step(θ, lj, lj_func, grad_func, τ, ϵ, std; rev_func=rev_func, log_func=log_func)
end


# TODO: remove used Turing-wrapper functions

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/base_hmc.hpp
function find_good_eps(model::Function, spl::Sampler{T}, vi::VarInfo) where T
    logpdf_func_float = gen_lj_func(vi, spl, model)

    spl.alg.gid != 0 && link!(vi, spl)

    @info "[Turing] looking for good initial eps..."
    ϵ = 0.1

    θ_dim = length(vi[spl])
    unitstd = ones(θ_dim)
    p = randn(θ_dim)

    H0 = _find_H(vi[spl], p, logpdf_func_float, unitstd)

    θ = vi[spl]
    θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
    h = τ == 0 ? Inf : _find_H(vi[spl], p_prime, logpdf_func_float, unitstd)

    delta_H = H0 - h
    direction = delta_H > log(0.8) ? 1 : -1

    iter_num = 1

    # Heuristically find optimal ϵ
    while (iter_num <= 12)

        p = randn(θ_dim)
        H0 = _find_H(vi[spl], p, logpdf_func_float, unitstd)

        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        h = τ == 0 ? Inf : _find_H(vi[spl], p_prime, logpdf_func_float, unitstd)
        @debug "direction = $direction, h = $h"

        delta_H = H0 - h

        if ((direction == 1) && !(delta_H > log(0.8)))
            break
        elseif ((direction == -1) && !(delta_H < log(0.8)))
            break
        else
            ϵ = direction == 1 ? 2.0 * ϵ : 0.5 * ϵ
        end

        iter_num += 1
    end

    while h == Inf  # revert if the last change is too big
        ϵ = ϵ / 2               # safe is more important than large
        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        h = τ == 0 ? Inf : _find_H(vi[spl], p_prime, logpdf_func_float, std)
    end
    @info "\r[$T] found initial ϵ: $ϵ"

    spl.alg.gid != 0 && invlink!(vi, spl)

    return ϵ
end

"""
    mh_accept(H::Real, H_new::Real)
    mh_accept(H::Real, H_new::Real, log_proposal_ratio::Real)

Peform MH accept criteria. Returns a boolean for whether or not accept and the
acceptance ratio in log space.
"""
mh_accept(H::Real, H_new::Real) = log(rand()) + H_new < min(H_new, H), min(0, -(H_new - H))
function mh_accept(H::Real, H_new::Real, log_proposal_ratio::Real)
    return log(rand()) + H_new < H + log_proposal_ratio, min(0, -(H_new - H))
end
