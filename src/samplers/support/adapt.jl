# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.

type WarmUpManager
  adapt_n   ::    Int
  params    ::    Dict
end

getindex(wum::WarmUpManager, param) = wum.params[param]

setindex!(wum::WarmUpManager, value, param) = wum.params[param] = value

init_warm_up_params{T<:Hamiltonian}(vi::VarInfo, spl::Sampler{T}) = begin
  wum = WarmUpManager(1, Dict())

  # Pre-cond
  reset_var_estimator(wum)
  wum[:stds] = 1.0

  # Dual averaging
  wum[:ϵ] = []
  reset_da(wum)
  wum[:n_warmup] = spl.alg.n_adapt
  wum[:δ] = spl.alg.delta

  # Stan.Adapt
  adapt_conf = spl.info[:adapt_conf]
  wum[:γ] = adapt_conf.gamma
  wum[:t_0] = adapt_conf.t0
  wum[:κ] = adapt_conf.kappa

  # Three phases settings
  # wum[:n_adapt] = spl.alg.n_adapt
  wum[:init_buffer] = adapt_conf.init_buffer
  wum[:term_buffer] = adapt_conf.term_buffer
  wum[:window_size] = adapt_conf.window
  wum[:next_window] = wum[:init_buffer] + wum[:window_size] - 1

  spl.info[:wum] = wum
end

reset_var_estimator(wum::WarmUpManager) = begin
  wum[:est_n]  = 0
  wum[:θ_mean] = 0.0  # zeros(D)
  wum[:M2]     = 0.0  # zeros(D)
end

reset_da(wum::WarmUpManager) = begin
  wum[:m] = 0
  wum[:x_bar] = 0.0
  wum[:H_bar] = 0.0
end

update_da_μ(wum::WarmUpManager, ϵ::Float64) = begin
  wum[:μ] = log(10 * ϵ)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
adapt_step_size(wum::WarmUpManager, stats::Float64) = begin

  dprintln(2, "adapting step size ϵ...")
  wum[:m] = wum[:m] + 1
  m = wum[:m]

  stats = stats > 1 ? 1 : stats

  γ = wum[:γ]; t_0 = wum[:t_0]; κ = wum[:κ]; δ = wum[:δ]
  μ = wum[:μ]; x_bar = wum[:x_bar]; H_bar = wum[:H_bar]

  H_η = 1.0 / (m + t_0)
  H_bar = (1.0 - H_η) * H_bar + H_η * (δ - stats)

  x = μ - H_bar * sqrt(m) / γ
  x_η = m^(-κ)
  x_bar = (1.0 - x_η) * x_bar + x_η * x

  ϵ = exp(x)
  
  push!(wum[:ϵ], ϵ)
  wum[:x_bar], wum[:H_bar] = x_bar, H_bar

  if m == wum[:n_warmup]
    dprintln(0, " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption.")
  end

end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
in_adaptation(wum::WarmUpManager) = (wum.adapt_n >= wum[:init_buffer]) && 
                                    (wum.adapt_n < wum[:n_warmup] - wum[:term_buffer]) && 
                                    (wum.adapt_n != wum[:n_warmup])

is_window_end(wum::WarmUpManager) = (wum.adapt_n == wum[:next_window]) && (wum.adapt_n != wum[:n_warmup])

compute_next_window(wum::WarmUpManager) = begin

  if ~(wum[:next_window] == wum[:n_warmup] - wum[:term_buffer] - 1)

  wum[:window_size] *= 2
  wum[:next_window] = wum.adapt_n + wum[:window_size]

    if ~(wum[:next_window] == wum[:n_warmup] - wum[:term_buffer] - 1)

      next_window_boundary = wum[:next_window] + 2 * wum[:window_size]

      if (next_window_boundary >= wum[:n_warmup] - wum[:term_buffer])
        wum[:next_window] = wum[:n_warmup] - wum[:term_buffer] - 1
      end

    end

  end

end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
update_pre_cond(wum::WarmUpManager, θ_new) = begin

  if in_adaptation(wum)

    wum[:est_n] = wum[:est_n] + 1

    delta = θ_new - wum[:θ_mean]
    wum[:θ_mean] = wum[:θ_mean] + delta / wum[:est_n]
    wum[:M2] = wum[:M2] + delta .* (θ_new - wum[:θ_mean])

  end

  if is_window_end(wum)

    compute_next_window(wum)

    # var = wum[:M2] / (wum[:est_n] - 1.0)
    # var = (wum[:est_n] / (wum[:est_n] + 5.0)) * var + 1e-3 * (5.0 / (wum[:est_n] + 5.0))

    # wum[:stds] = sqrt.(var)

    reset_var_estimator(wum)

    return true

  end

  return false

end

adapt(wum::WarmUpManager, stats::Float64, θ_new) = begin

  if wum.adapt_n <= wum[:n_warmup]
    adapt_step_size(wum, stats)
    is_update = update_pre_cond(wum, θ_new)
    wum.adapt_n += 1

    if is_update
      update_da_μ(wum, mean(wum[:ϵ][end-20:end]))
      reset_da(wum)
    end

  end

end
