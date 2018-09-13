# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp

"""
    gen_grad_func(vi, spl, model)

Generate a function that takes `θ` and returns log-joint probabilty and gradient at `θ`,
using Turing-related variables `vi`, `spl` and `model`.
"""
function gen_grad_func(vi, spl, model)

    grad_func(θ::T) where {T<:Union{Vector,SubArray}} = begin
        if ADBACKEND[] == :forward_diff
            lp, grad = gradient_forward(θ, vi, model, spl)
        elseif ADBACKEND[] == :reverse_diff
            lp, grad = gradient_reverse(θ, vi, model, spl)
        end
    return lp, grad
  end

  return grad_func
end

"""
    gen_lj_func(vi, spl, model)

Generate a function that takes `θ` and returns log-joint probabilty at `θ`, 
using Turing-related variables `vi`, `spl` and `model`.
"""
function gen_lj_func(vi, spl, model)

    lj_func(θ) = begin
        vi[spl][:] = θ[:]
        return runmodel!(model, vi, spl).logp
    end

    return lj_func
end

"""
    gen_rev_func(vi, spl)

Generate a function that takes `x_old` and `lp_old` and resets the values in `vi` for 
`spl` as `x_old` and the log-joint probabilty as `lp_old`, using local 
variables `vi` and `spl` in the scope where `gen_rev_func` is called.
"""
function gen_rev_func(vi, spl)

    rev_func(θ_old::T, old_logp::R) where {T<:Union{Vector,SubArray},R<:Real} = begin
        if ADBACKEND[] == :forward_diff
            vi[spl] = θ_old
        elseif ADBACKEND[] == :reverse_diff
            vi_spl = vi[spl]
            for i = 1:length(θ_old)
                vi_spl[i] = θ_old[i]
            end
        else
            @error("Unsupported ADBACKEND = $(ADBACKEND[])")
        end
        setlogp!(vi, old_logp)
        return vi.logp
  end

  return rev_func
end

"""
    gen_log_func(spl)

Generate a function that takes no argument and performs logging for number of leapfrog 
steps used, using the local variable `spl` in the scope where `gen_log_func` is called.
"""
function gen_log_func(spl)

    log_func() = begin
        spl.info[:lf_num] += 1
        spl.info[:total_lf_num] += 1  # record leapfrog num
    end

    return log_func
end

function runmodel!(model::Function, vi::VarInfo, spl::Union{Nothing,Sampler}) 
    setlogp!(vi, zero(Real))
    if spl != nothing 
        spl.info[:total_eval_num] += 1 
    end
    # model(vi=vi, sampler=spl) # run model
    Base.invokelatest(model, vi, spl)
    vi
end

function sample_momentum(vi::VarInfo, spl::Sampler)
    @debug "sampling momentum..."
    # randn(length(getranges(vi, spl))) ./ spl.info[:wum][:stds]

    d = length(getranges(vi, spl))
    stds = spl.info[:wum][:stds]

    return _sample_momentum(d, stds)
end

function _sample_momentum(d::Int, stds::Vector)
    return randn(d) ./ stds
end

# Leapfrog step
# NOTE: leapfrog() doesn't change θ in place!
function leapfrog(
         _θ::T, 
         p::Vector{Float64}, 
         τ::Int, 
         ϵ::Float64,
         model::Function, 
         vi::VarInfo, 
         spl::Sampler
        ) where {T<:Union{Vector,SubArray}}

    θ = realpart(_θ)

    lp_grad_func = gen_grad_func(vi, spl, model)
    rev_func = gen_rev_func(vi, spl)
    log_func = gen_log_func(spl)

    return _leapfrog(θ, p, τ, ϵ, lp_grad_func; rev_func=rev_func, log_func=log_func)
end

function _leapfrog(
                   θ::T, 
                   p::Vector{Float64}, 
                   τ::Int, 
                   ϵ::Float64, 
                   lp_grad_func::Function;
                   rev_func=nothing, 
                   log_func=nothing) where {T<:Union{Vector,SubArray}}

    old_logp, grad = lp_grad_func(θ)
    verifygrad(grad) || (return θ, p, 0)

    τ_valid = 0
    for t in 1:τ
        # NOTE: we dont need copy here becase arr += another_arr
        #       doesn't change arr in-place
        p_old = p; 
        θ_old = copy(θ)

        p -= ϵ .* grad / 2
        θ += ϵ .* p  # full step for state

        if log_func != nothing 
            log_func() 
        end

        old_logp, grad = lp_grad_func(θ)
        if ~verifygrad(grad)
            if rev_func != nothing 
                rev_func(θ_old, old_logp) 
            end
            θ = θ_old; 
            p = p_old; 
            break
        end

        p -= ϵ * grad / 2
        τ_valid += 1
    end

    θ, p, τ_valid
end

# Compute Hamiltonian
function find_H(p::Vector, model::Function, vi::VarInfo, spl::Sampler)

  # Old code
  # # NOTE: getlogp(vi) = 0 means the current vals[end] hasn't been used at all.
  # #       This can be a result of link/invlink (where expand! is used)
  # if getlogp(vi) == 0 vi = runmodel(model, vi, spl) end
  #
  # p_orig = p .* spl.info[:wum][:stds]
  #
  # H = dot(p_orig, p_orig) / 2 + realpart(-getlogp(vi))
  # if isnan(H) H = Inf else H end
  #
  # H

    logpdf_func_float = gen_lj_func(vi, spl, model)
    return _find_H(vi[spl], p, logpdf_func_float, spl.info[:wum][:stds])
end

function _find_H(
                 theta::T, 
                 p::Vector, 
                 logpdf_func_float::Function, 
                 stds::Vector
                ) where {T<:Union{Vector,SubArray}}

    lp = logpdf_func_float(theta)
    return _find_H(theta, p, lp, stds)
end

function _find_H(
                 theta::T, 
                 p::Vector, 
                 lp::Real, 
                 stds::Vector
                ) where {T<:Union{Vector,SubArray}}

  p_orig = p .* stds
  H = 0.5 * dot(p_orig, p_orig) + (-lp)
  H = isnan(H) ? Inf : H

  return H
end

# TODO: remove used Turing-wrapper functions

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/base_hmc.hpp
function find_good_eps(model::Function, vi::VarInfo, spl::Sampler{T}) where T
    @info("[Turing] looking for good initial eps...")

    ϵ = 0.1

    p = sample_momentum(vi, spl)
    H0 = find_H(p, model, vi, spl)

    θ = realpart(vi[spl])
    θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
    h = τ == 0 ? Inf : find_H(p_prime, model, vi, spl)

    delta_H = H0 - h
    direction = delta_H > log(0.8) ? 1 : -1

    iter_num = 1

    # Heuristically find optimal ϵ
    while (iter_num <= 12)

        p = sample_momentum(vi, spl)
        H0 = find_H(p, model, vi, spl)

        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        h = τ == 0 ? Inf : find_H(p_prime, model, vi, spl)
        @debug "direction = $direction, h = $h"

        delta_H = H0 - h

        if ((direction == 1) && !(delta_H > log(0.8)))
            break;
        elseif ((direction == -1) && !(delta_H < log(0.8)))
            break;
        else
            ϵ = direction == 1 ? 2.0 * ϵ : 0.5 * ϵ
        end

        iter_num += 1
    end

    while h == Inf  # revert if the last change is too big
        ϵ = ϵ / 2               # safe is more important than large
        θ_prime, p_prime, τ = leapfrog(θ, p, 1, ϵ, model, vi, spl)
        h = τ == 0 ? Inf : find_H(p_prime, model, vi, spl)
    end
    @info("\r[$T] found initial ϵ: ", ϵ)
    return ϵ
end

"""
    mh_accept(H, H_new)

Peform MH accept criteria. Returns a boolean for whether or not accept and the 
acceptance ratio in log space.

"""
function mh_accept(H, H_new; log_proposal_ratio=nothing)

    logα = min(0, -(H_new - H))

    u = rand()
    logu = log(u)

    if log_proposal_ratio == nothing
        is_accept = (logu + H_new < min(H_new, H))
    else
        is_accept = (logu + H_new < H + log_proposal_ratio)
    end
    return is_accept, logα
end
