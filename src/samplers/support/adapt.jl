init_warm_up_params{T<:Hamiltonian}(vi::VarInfo, spl::Sampler{T}) = begin
  # Pre-cond
  spl.info[:θ_mean] = realpart(vi[spl])
  spl.info[:θ_num] = 1
  D = length(vi[spl])
  spl.info[:stds] = ones(D)
  spl.info[:θ_vars] = ones(D)
  # DA
  if ~haskey(spl.info, :ϵ)
    spl.info[:ϵ] = nothing
  end
  spl.info[:μ] = nothing
  spl.info[:ϵ_bar] = 1.0
  spl.info[:H_bar] = 0.0
  spl.info[:m] = 0
end

update_da_params{T<:Hamiltonian}(spl::Sampler{T}, ϵ::Float64) = begin
  spl.info[:ϵ] = [ϵ]
  spl.info[:μ] = log(10 * ϵ)
end

adapt_step_size{T<:Hamiltonian}(spl::Sampler{T}, stats::Float64, δ::Float64) = begin
  dprintln(2, "adapting step size ϵ...")
  m = spl.info[:m] += 1
  if m <= spl.alg.n_adapt
    γ = 0.05; t_0 = 10; κ = 0.75
    μ = spl.info[:μ]; ϵ_bar = spl.info[:ϵ_bar]; H_bar = spl.info[:H_bar]

    H_bar = (1 - 1 / (m + t_0)) * H_bar + 1 / (m + t_0) * (δ - stats)
    ϵ = exp(μ - sqrt(m) / γ * H_bar)
    dprintln(1, " ϵ = $ϵ, stats = $stats")

    ϵ_bar = exp(m^(-κ) * log(ϵ) + (1 - m^(-κ)) * log(ϵ_bar))
    push!(spl.info[:ϵ], ϵ)
    spl.info[:ϵ_bar], spl.info[:H_bar] = ϵ_bar, H_bar

    if m == spl.alg.n_adapt
      dprintln(0, " Adapted ϵ = $ϵ, $m HMC iterations is used for adaption.")
    end
  end
end

update_pre_cond{T<:Hamiltonian}(vi::VarInfo, spl::Sampler{T}) = begin
  θ_new = realpart(vi[spl])                                         # x_t
  spl.info[:θ_num] += 1
  t = spl.info[:θ_num]                                              # t
  θ_mean_old = copy(spl.info[:θ_mean])                              # x_bar_t-1
  spl.info[:θ_mean] = (t - 1) / t * spl.info[:θ_mean] + θ_new / t   # x_bar_t
  θ_mean_new = spl.info[:θ_mean]                                    # x_bar_t

  if t == 2
    first_two = [θ_mean_old'; θ_new'] # θ_mean_old here only contains the first θ
    spl.info[:θ_vars] = diag(cov(first_two))
  elseif t <= 1000
    D = length(θ_new)
    # D = 2.4^2
    spl.info[:θ_vars] = (t - 1) / t * spl.info[:θ_vars] .+ 100 * eps(Float64) +
                        (2.4^2 / D) / t * (t * θ_mean_old .* θ_mean_old - (t + 1) * θ_mean_new .* θ_mean_new + θ_new .* θ_new)
  end

  if t > 500
    spl.info[:stds] = sqrt(spl.info[:θ_vars])
    spl.info[:stds] = spl.info[:stds] / min(spl.info[:stds]...)
  end
end



type WarmUpManager
  state       ::    Int
  curr_iter   ::    Int
  info        ::    Dict
end

update_state(wum::WarmUpManager) = begin

end
