type WarmUpManager
  iter_n    ::    Int
  state     ::    Int
  params    ::    Dict
end

getindex(wum::WarmUpManager, param) = wum.params[param]

setindex!(wum::WarmUpManager, value, param) = wum.params[param] = value

init_warm_up_params{T<:Hamiltonian}(vi::VarInfo, spl::Sampler{T}) = begin
  wum = WarmUpManager(1, 1, Dict())

  # Pre-cond
  wum[:θ_num] = 0
  wum[:θ_mean] = nothing
  D = length(vi[spl])
  wum[:stds] = ones(D)
  wum[:vars] = ones(D)

  # DA
  wum[:ϵ] = nothing
  wum[:μ] = nothing
  wum[:ϵ_bar] = 1.0
  wum[:H_bar] = 0.0
  wum[:m] = 0
  wum[:n_adapt] = spl.alg.n_adapt
  wum[:δ] = spl.alg.delta

  # Stan.Adapt
  adapt_conf = spl.info[:adapt_conf]
  wum[:γ] = adapt_conf.gamma
  wum[:t_0] = adapt_conf.t0
  wum[:κ] = adapt_conf.kappa

  spl.info[:wum] = wum
end

update_da_params(wum::WarmUpManager, ϵ::Float64) = begin
  wum[:ϵ] = [ϵ]
  wum[:μ] = log(10 * ϵ)
end

adapt_step_size(wum::WarmUpManager, stats::Float64) = begin
  dprintln(2, "adapting step size ϵ...")
  m = wum[:m] += 1
  if m <= wum[:n_adapt]
    γ = wum[:γ]; t_0 = wum[:t_0]; κ = wum[:κ]; δ = wum[:δ]
    μ = wum[:μ]; ϵ_bar = wum[:ϵ_bar]; H_bar = wum[:H_bar]

    H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - stats)
    ϵ = exp(μ - sqrt(m) / γ * H_bar)
    dprintln(1, " ϵ = $ϵ, stats = $stats")

    ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
    push!(wum[:ϵ], ϵ)
    wum[:ϵ_bar], wum[:H_bar] = ϵ_bar, H_bar

    if m == wum[:n_adapt]
      dprintln(0, " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption.")
    end
  end
end

update_pre_cond(wum::WarmUpManager, θ_new) = begin

  wum[:θ_num] += 1                                      # θ_new = x_t
  t = wum[:θ_num]                                       # t

  if t == 1
    wum[:θ_mean] = θ_new
  else
    θ_mean_old = copy(wum[:θ_mean])                       # x_bar_t-1
    wum[:θ_mean] = (t - 1) / t * wum[:θ_mean] + θ_new / t # x_bar_t
    θ_mean_new = wum[:θ_mean]                             # x_bar_t

    if t == 2
      first_two = [θ_mean_old'; θ_new'] # θ_mean_old here only contains the first θ
      wum[:vars] = diag(cov(first_two))
    else#if t <= 1000
      D = length(θ_new)
      # D = 2.4^2
      wum[:vars] = (t - 1) / t * wum[:vars] .+ 1e3 * eps(Float64) +
                          (2.4^2 / D) / t * (t * θ_mean_old .* θ_mean_old - (t + 1) * θ_mean_new .* θ_mean_new + θ_new .* θ_new)
    end

    if t > 100
      wum[:stds] = sqrt(wum[:vars])
      # wum[:stds] = wum[:stds] / min(wum[:stds]...)  # old
      wum[:stds] = wum[:stds] / mean([wum[:stds]...])
    end
  end
end

update_state(wum::WarmUpManager) = begin
  # TODO: make use of Stan.Adapt.init_buffer, Stan.Adapt.term_buffer and Stan.Adapt.window
  wum.iter_n += 1   # update iteration number

  # Update state
  if wum.state == 1
    if wum.iter_n > 100
      wum.state = 2
    end
  elseif wum.state == 2
    if wum.iter_n > 900
      wum.state = 3
    end
  elseif wum.state == 3
    if wum.iter_n > 1000
      wum.state = 4
    end
  elseif wum.state == 4
    # no more change
  else
    error("[Turing.WarmUpManager] unknown state $(wum.state)")
  end
end

adapt(wum::WarmUpManager, stats::Float64, θ_new) = begin
  update_state(wum)

  # Use Dual Averaging to adapt ϵ
  if wum.state in [1, 2, 3]
    adapt_step_size(wum, stats)
  end

  # Update pre-conditioning matrix
  if wum.state == 2
    update_pre_cond(wum, θ_new)
  end
end
