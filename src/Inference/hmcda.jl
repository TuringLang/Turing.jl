function hmc_step(θ, lj, lj_func, grad_func, H_func, ϵ, alg::HMCDA, momentum_sampler::Function;
                  rev_func=nothing, log_func=nothing)
    θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(
                θ, lj, lj_func, grad_func, H_func, ϵ, alg.lambda, momentum_sampler; rev_func=rev_func, log_func=log_func)
    return θ_new, lj_new, is_accept, α
end
