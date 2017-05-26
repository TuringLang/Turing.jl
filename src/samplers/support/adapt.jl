adapt_step_size(spl::Sampler, stats::Float64) = begin
  dprintln(2, "adapting step size ϵ...")
  m = spl.info[:m] += 1
  if m <= spl.alg.n_adapt
    δ = spl.alg.delta
    μ = spl.info[:μ]
    γ = 0.05
    t_0 = 10
    κ = 0.75
    ϵ_bar = spl.info[:ϵ_bar]
    H_bar = spl.info[:H_bar]


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

macro init_da_parameters()
  quote
    spl.info[:ϵ] = [ϵ]
    spl.info[:μ] = log(10 * ϵ)
    spl.info[:ϵ_bar] = 1.0
    spl.info[:H_bar] = 0.0
    spl.info[:m] = 0
  end
end
