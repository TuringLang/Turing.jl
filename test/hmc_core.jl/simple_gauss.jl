using Distributions, DiffBase
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
using Turing: _sample_momentum, _leapfrog, _find_H

# @model gdemo(x) = begin

#     s ~ InverseGamma(2,3)
#     m ~ Normal(0,sqrt(s))
#     x[1] ~ Normal(m, sqrt(s))
#     x[2] ~ Normal(m, sqrt(s))
    
#     return s, m

# end

θ_dim = 1
function lj_func(θ)
  _lj = zero(Real)
  
  s = 1

  m = θ[1]
  _lj += logpdf(Normal(0, sqrt(s)), m)

  _lj += logpdf(Normal(m, sqrt(s)), 2.0)
  _lj += logpdf(Normal(m, sqrt(s)), 2.5)

  return _lj
end

neg_lj_func(θ) = -lj_func(θ)
const f_tape = GradientTape(neg_lj_func, randn(θ_dim))
const compiled_f_tape = compile(f_tape)

function grad_func(θ)
    
  inputs = θ
  results = similar(θ)
  all_results = DiffBase.GradientResult(results)

  gradient!(all_results, compiled_f_tape, inputs)

  neg_lj = all_results.value
  grad, = all_results.derivs

  return -neg_lj, grad

end

function _hmc_step(θ, lj, lj_func, grad_func, ϵ::Float64, λ::Float64, stds;
                   dprint=dprintln)

  θ_dim = length(θ)

  dprint(2, "sampling momentums...")
  p = _sample_momentum(θ_dim, stds)

  dprint(2, "recording old values...")
  H = _find_H(θ, p, lj, stds)

  τ = max(1, round(Int, λ / ϵ))
  dprint(2, "leapfrog for $τ steps with step size $ϵ")
  θ_new, p_new, τ_valid = _leapfrog(θ, p, τ, ϵ, grad_func)

  dprint(2, "computing new H...")
  lj_new = lj_func(θ_new)
  H_new = (τ_valid == 0) ? Inf : _find_H(θ_new, p_new, lj_new, stds)

  dprint(2, "computing accept rate α...")
  α = min(1, exp(-(H_new - H)))

  # if PROGRESS && spl.alg.gid == 0
  # stds_str = string(spl.info[:wum][:stds])
  # stds_str = length(stds_str) >= 32 ? stds_str[1:30]*"..." : stds_str
  # haskey(spl.info, :progress) && ProgressMeter.update!(
  #                                     spl.info[:progress],
  #                                     spl.info[:progress].counter; showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, stds_str)]
  #                                 )
  # end

  dprint(2, "decide wether to accept...")
  is_accept = false
  if rand() < α             # accepted
    θ = θ_new
    lj = lj_new
    is_accept = true
  # push!(spl.info[:accept_his], true)
  else                      # rejected
  # push!(spl.info[:accept_his], false)

  # Reset Θ
  # NOTE: ForwardDiff and ReverseDiff need different implementation
  #       due to immutable Dual vs mutable TrackedReal
  # if ADBACKEND == :forward_diff

  #     vi[spl] = old_θ

  # elseif ADBACKEND == :reverse_diff

  #     vi_spl = vi[spl]
  #     for i = 1:length(old_θ)
  #     if isa(vi_spl[i], ReverseDiff.TrackedReal)
  #         vi_spl[i].value = old_θ[i]
  #     else
  #         vi_spl[i] = old_θ[i]
  #     end
  #     end

  # end

  # setlogp!(vi, old_logp)  # reset logp
  end


  # if spl.alg.delta > 0 && τ_valid > 0    # only do adaption for HMCDA
  # adapt!(spl.info[:wum], α, realpart(vi[spl]), adapt_ϵ = true)
  # end

  dprint(3, "R -> X...")
  # if spl.alg.gid != 0 invlink!(vi, spl); cleandual!(vi) end

  return θ, lj, is_accept

end

_hmc_step(θ, lj, lj_func, grad_func, τ::Int, ϵ::Float64, stds; dprint=dprintln) =
  _hmc_step(θ, lj, lj_func, grad_func, ϵ, τ * ϵ, stds; dprint=dprint)

# @show _hmc_step(θ, lj, lj_func, grad_func, 5, 0.01, stds)


stds = ones(θ_dim)
θ = randn(θ_dim)
lj = lj_func(θ)

chn = []
accept_num = 1

function dummy_print(args...)
  nothing
end

totla_num = 5000
for iter = 1:totla_num
  push!(chn, θ)
  θ, lj, is_accept = _hmc_step(θ, lj, lj_func, grad_func, 5, 0.05, stds; dprint=dummy_print)
  accept_num += is_accept
  if (iter % 50 == 0) println(θ) end
end

@show mean(chn), lj
@show accept_num / totla_num