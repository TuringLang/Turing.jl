# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp

using Statistics: middle

"""
    gen_grad_func(vi::VarInfo, sampler::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, sampler, model)`.
"""
function gen_grad_func(vi::VarInfo, sampler::Sampler, model)
    return θ::AbstractVector{<:Real}->gradient_logp(θ, vi, model, sampler)
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

# TODO: improve typing for all generator functions

function gen_momentum_sampler(vi::VarInfo, spl::Sampler)
    d = length(vi[spl])
    return function()
        return randn(d)
    end
end

function gen_H_func()
    return function(θ::AbstractVector{<:Real},
                    p::AbstractVector{<:Real},
                    logp::Real)
        H = sum(abs2, p) / 2 - logp
        return isnan(H) ? Inf : H
    end
end

function gen_momentum_sampler(vi::VarInfo, spl::Sampler, ::UnitPreConditioner)
    d = length(vi[spl])
    return function()
        return randn(d)
    end
end

function gen_H_func(::UnitPreConditioner)
    return function(θ::AbstractVector{<:Real},
                    p::AbstractVector{<:Real},
                    logp::Real)
        H = sum(abs2, p) / 2 - logp
        return isnan(H) ? Inf : H
    end
end

function gen_momentum_sampler(vi::VarInfo, spl::Sampler, pc::DiagPreConditioner)
    d = length(vi[spl])
    std = pc.std
    return function()
        return randn(d) ./ std
    end
end

function gen_H_func(pc::DiagPreConditioner)
    std = pc.std
    return function(θ::AbstractVector{<:Real},
                    p::AbstractVector{<:Real},
                    logp::Real)
        H = sum(abs2, p .* std) / 2 - logp
        return isnan(H) ? Inf : H
    end
end

# NOTE: related Hamiltonian change: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp
function gen_momentum_sampler(vi::VarInfo, spl::Sampler, pc::DensePreConditioner)
    d = length(vi[spl])
    A = Symmetric(pc.covar)
    C = LinearAlgebra.cholesky(A)
    return function()
        return C.U \ randn(d)
    end
end

function gen_H_func(pc::DensePreConditioner)
    A = pc.covar
    return function(θ::AbstractVector{<:Real},
                    p::AbstractVector{<:Real},
                    logp::Real)
        H = p' * A * p / 2 - logp
        return isnan(H) ? Inf : H
    end
end

function leapfrog(θ::AbstractVector{<:Real},
                  p::AbstractVector{<:Real},
                  τ::Int,
                  ϵ::Real,
                  model,
                  vi::VarInfo,
                  sampler::Sampler,
                  )
    lp_grad = gen_grad_func(vi, sampler, model)
    rev = gen_rev_func(vi, sampler)
    logger = gen_log_func(sampler)
    return _leapfrog(θ, p, τ, ϵ, lp_grad; rev_func=rev, log_func=logger)
end

function _leapfrog(θ::AbstractVector{<:Real},
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

    p .+= ϵ .* grad ./ 2
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

        p .+= ϵ .* grad
        τ_valid += 1
    end

    # Undo half a step in the momenta.
    p .-= ϵ .* grad ./ 2

    return θ, p, τ_valid
end

function _hmc_step(θ::AbstractVector{<:Real},
                   lj::Real,
                   lj_func::Function,
                   grad_func::Function,
                   H_func::Function,
                   τ::Int,
                   ϵ::Real,
                   momentum_sampler::Function;
                   rev_func=nothing,
                   log_func=nothing,
                   )
    Turing.DEBUG && @debug "sampling momentums..."
    p = momentum_sampler()

    Turing.DEBUG && @debug "recording old values..."
    H = H_func(θ, p, lj)

    Turing.DEBUG && @debug "leapfrog for $τ steps with step size $ϵ"
    θ_new, p_new, τ_valid = _leapfrog(θ, p, τ, ϵ, grad_func; rev_func=rev_func, log_func=log_func)

    Turing.DEBUG && @debug "computing new H..."
    lj_new = lj_func(θ_new)
    H_new = (τ_valid == 0) ? Inf : H_func(θ_new, p_new, lj_new)

    Turing.DEBUG && @debug "deciding wether to accept and computing accept rate α..."
    is_accept, logα = mh_accept(H, H_new)

    if is_accept
        θ = θ_new
        lj = lj_new
    end

    return θ, lj, is_accept, τ_valid, exp(logα)
end

function _hmc_step(θ::AbstractVector{<:Real},
                   lj::Real,
                   lj_func::Function,
                   grad_func::Function,
                   H_func::Function,
                   ϵ::Real,
                   λ::Real,
                   momentum_sampler::Function;
                   rev_func=nothing,
                   log_func=nothing,
                   )
    τ = max(1, round(Int, λ / ϵ))
    return _hmc_step(θ, lj, lj_func, grad_func, H_func, τ, ϵ, momentum_sampler;
                     rev_func=rev_func, log_func=log_func)

end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/base_hmc.hpp
function find_good_eps(model, spl::Sampler{T}, vi::VarInfo) where T
    lj_func = gen_lj_func(vi, spl, model)
    momentum_sampler = gen_momentum_sampler(vi, spl)
    grad_func = gen_grad_func(vi, spl, model)
    H_func = gen_H_func()
    θ, lj = vi[spl], vi.logp
    ϵ = _find_good_eps(θ, lj_func, grad_func, H_func, momentum_sampler)
    vi[spl] = θ
    setlogp!(vi, lj)
    @info "[Turing] found initial ϵ: $ϵ"
    return ϵ
end

##
## Heuristically find optimal ϵ
##
function _find_good_eps(θ, lj_func, grad_func, H_func, momentum_sampler; max_num_iters=100)
    @info "[Turing] looking for good initial eps..."
    ϵ_prime = ϵ = 0.1
    a_min, a_cross, a_max = 0.25, 0.5, 0.75 # minimal, crossing, maximal accept ratio
    d = 2.0

    p = momentum_sampler()
    H0 = H_func(θ, p, lj_func(θ))

    θ_prime, p_prime, τ = _leapfrog(θ, p, 1, ϵ, grad_func)
    h = τ == 0 ? Inf : H_func(θ_prime, p_prime, lj_func(θ_prime))

    delta_H = H0 - h # logp(θ`) - logp(θ)
    direction = delta_H > log(a_cross) ? 1 : -1

    # Crossing step: increase/decrease ϵ until accept ratio cross a_cross.
    for _ = 1:max_num_iters
        ϵ_prime = direction == 1 ? d * ϵ : 1/d * ϵ
        θ_prime, p_prime, τ = _leapfrog(θ, p, 1, ϵ_prime, grad_func)
        h = τ == 0 ? Inf : H_func(θ_prime, p_prime, lj_func(θ_prime))
        Turing.DEBUG && @debug "direction = $direction, h = $h"

        delta_H = H0 - h

        Turing.DEBUG && @debug "[Turing] ϵ = $ϵ_prime, accept ratio a = $(min(1,(exp(delta_H))))"
        if ((direction == 1) && !(delta_H > log(a_cross)))
            break
        elseif ((direction == -1) && !(delta_H < log(a_cross)))
            break
        else
            ϵ = ϵ_prime
        end
    end

    # Bisection step: ensure final accept ratio:  a_min < a < a_max.
    #  See https://en.wikipedia.org/wiki/Bisection_method
    ϵ, ϵ_prime = ϵ < ϵ_prime ? (ϵ, ϵ_prime) : (ϵ_prime, ϵ) # Ensure ϵ < ϵ_prime
    for _ = 1:max_num_iters
        ϵ_mid = middle(ϵ, ϵ_prime)
        θ_prime, p_prime, τ = _leapfrog(θ, p, 1, ϵ_mid, grad_func)
        h = τ == 0 ? Inf : H_func(θ_prime, p_prime, lj_func(θ_prime))

        delta_H = H0 - h

        Turing.DEBUG && @debug "[Turing] ϵ = $ϵ_mid, accept ratio a = $(min(1,(exp(delta_H))))"
        if (exp(delta_H) > a_max)
            ϵ = ϵ_mid
        elseif (exp(delta_H) < a_min)
            ϵ_prime = ϵ_mid
        else
            ϵ = ϵ_mid; break
        end
    end

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
