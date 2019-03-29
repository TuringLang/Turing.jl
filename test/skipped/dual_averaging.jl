function _adapt_ϵ(logϵ, Hbar, logϵbar, da_stat, m, M_adapt, δ, μ;
    γ=0.05, t0=10, κ=0.75)

    if m <= M_adapt
        Hbar = (1.0 - 1.0 / (m + t0)) * Hbar + (1 / (m + t0)) * (δ - da_stat)
        logϵ = μ - sqrt(m) / γ * Hbar
        logϵbar = m^(-κ) * logϵ + (1 - m^(-κ)) * logϵbar
    else
        logϵ = logϵbar
    end
    return logϵ, Hbar, logϵbar
end
