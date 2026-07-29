#
# Exact inference for the two state space models with tractable posteriors
#
# These give samplers something unarguable to be compared against: a Kalman filter and smoother for
# a scalar linear Gaussian model, and forward-backward for a discrete HMM. `test_exact_ssm_reference`
# validates both against brute force, so the reference is checked in CI rather than trusted.
#

module ExactSSM

using Distributions: Categorical, MvNormal, Normal, logpdf
using LinearAlgebra: I, Symmetric, diag, eigen
using Test: @test, @testset

# Everything here is reached qualified (`ExactSSM.foo`), so there is no export list.

##
## Scalar linear Gaussian SSM: x₁ ~ N(0, s0), xₜ = a·xₜ₋₁ + N(0, q), yₜ = xₜ + N(0, r)
##

# Stacking x = x₁:T gives x = Lε with ε ~ N(0, D), L unit-lower-triangular from the AR(1) recursion
# and D = diag(s0, q, …, q). So Σx = L D Lᵀ and Σy = Σx + rI, and everything follows by conditioning
# a jointly Gaussian vector -- closed form, with no recursion to get subtly wrong.

"Joint prior covariance of `x₁:T` for the AR(1) latent chain."
function lgssm_prior_cov(T::Integer, a::Real, q::Real, s0::Real)
    F = typeof(one(a) * one(q) * one(s0))
    L = [j <= i ? F(a)^(i - j) : zero(F) for i in 1:T, j in 1:T]
    D = fill(F(q), T)
    D[1] = s0
    return Symmetric(L * (D .* L'))
end

"""
    lgssm_smoother(y, a, q, r, s0) -> (means, variances)

Exact smoothing marginals `E[xₜ | y₁:T]` and `Var[xₜ | y₁:T]`.
"""
function lgssm_smoother(y::AbstractVector, a::Real, q::Real, r::Real, s0::Real)
    Σx = lgssm_prior_cov(length(y), a, q, s0)
    G = Σx / (Σx + r * I)                     # Σx Σy⁻¹; both means are zero a priori
    return G * y, diag(Symmetric(Σx - G * Σx))
end

"Exact marginal log-likelihood `log p(y₁:T)`."
function lgssm_loglik(y::AbstractVector, a::Real, q::Real, r::Real, s0::Real)
    Σy = lgssm_prior_cov(length(y), a, q, s0) + r * I
    return logpdf(MvNormal(zeros(eltype(Σy), length(y)), Σy), y)
end

"Kalman filter plus RTS smoother, kept only to cross-check the closed forms above."
function lgssm_kalman(y::AbstractVector, a::Real, q::Real, r::Real, s0::Real)
    T = length(y)
    F = typeof(one(a) * one(q) * one(r) * one(s0) * one(eltype(y)))
    mp, Pp, mf, Pf = (zeros(F, T) for _ in 1:4)
    ll = zero(F)
    for t in 1:T
        mp[t] = t == 1 ? zero(F) : a * mf[t - 1]
        Pp[t] = t == 1 ? F(s0) : a^2 * Pf[t - 1] + q
        S = Pp[t] + r
        ll += -(log(2pi * S) + (y[t] - mp[t])^2 / S) / 2
        K = Pp[t] / S
        mf[t] = mp[t] + K * (y[t] - mp[t])
        Pf[t] = (1 - K) * Pp[t]
    end
    ms, Ps = copy(mf), copy(Pf)
    for t in (T - 1):-1:1
        C = Pf[t] * a / Pp[t + 1]
        ms[t] = mf[t] + C * (ms[t + 1] - mp[t + 1])
        Ps[t] = Pf[t] + C^2 * (Ps[t + 1] - Pp[t + 1])
    end
    return ms, Ps, ll
end

##
## Discrete HMM: z₁ ~ Categorical(π0), zₜ | zₜ₋₁ ~ Categorical(P[zₜ₋₁, :]), yₜ | zₜ
##

"""
    hmm_forward_backward(π0, P, loglik_obs) -> (posterior, loglik)

Forward-backward, where `loglik_obs[t, k] = log p(yₜ | zₜ = k)`. `posterior[t, k]` is
`p(zₜ = k | y₁:T)`.
"""
function hmm_forward_backward(
    π0::AbstractVector, P::AbstractMatrix, loglik_obs::AbstractMatrix
)
    T, K = size(loglik_obs)
    F = promote_type(eltype(π0), eltype(P), eltype(loglik_obs))
    lik = exp.(loglik_obs)                     # both passes need it; exponentiate once
    α = zeros(F, T, K)
    c = zeros(F, T)                            # per-step normalisers, which give the log-likelihood
    α[1, :] = π0 .* @view lik[1, :]
    c[1] = sum(@view α[1, :])
    α[1, :] ./= c[1]
    for t in 2:T
        α[t, :] = (P' * @view(α[t - 1, :])) .* @view lik[t, :]
        c[t] = sum(@view α[t, :])
        α[t, :] ./= c[t]
    end
    β = ones(F, T, K)
    for t in (T - 1):-1:1
        β[t, :] = P * (@view(lik[t + 1, :]) .* @view(β[t + 1, :])) ./ c[t + 1]
    end
    post = α .* β
    return post ./ sum(post; dims=2), sum(log, c)
end

"Brute-force HMM posterior and log-likelihood by enumerating all `K^T` state paths."
function hmm_brute_force(π0::AbstractVector, P::AbstractMatrix, loglik_obs::AbstractMatrix)
    T, K = size(loglik_obs)
    post = zeros(T, K)
    total = 0.0
    for z in CartesianIndices(ntuple(_ -> K, T))
        lp = log(π0[z[1]]) + loglik_obs[1, z[1]]
        for t in 2:T
            lp += log(P[z[t - 1], z[t]]) + loglik_obs[t, z[t]]
        end
        w = exp(lp)
        total += w
        for t in 1:T
            post[t, z[t]] += w
        end
    end
    return post ./ total, log(total)
end

##
## Shared helpers
##

"Stationary distribution of a row-stochastic transition matrix."
function stationary_distribution(P::AbstractMatrix)
    ev = eigen(collect(transpose(P)))
    π0 = real.(ev.vectors[:, argmin(abs.(ev.values .- 1))])
    return π0 ./ sum(π0)
end

"""
    grid_posterior(prior, θs, loglik) -> weights

Normalised posterior weights over a parameter grid, from `p(θ | y) ∝ p(θ)·p(y | θ)`. Given an exact
`loglik`, this makes the θ posterior exact rather than another Monte Carlo estimate.
"""
function grid_posterior(prior, θs, loglik)
    logw = [logpdf(prior, θ) + loglik(θ) for θ in θs]
    w = exp.(logw .- maximum(logw))
    return w ./ sum(w)
end

"Mean and standard deviation of a grid posterior."
function grid_moments(w::AbstractVector, θs)
    m = sum(w .* θs)
    return m, sqrt(sum(w .* (θs .- m) .^ 2))
end

"""
Check the exact implementations against brute force, so that anything comparing a sampler to them is
comparing against something independently verified. The Gaussian closed form is checked against a
Kalman recursion, and forward-backward against enumeration of every state path.
"""
function test_exact_ssm_reference()
    @testset "exact SSM reference" begin
        a, q, r, s0, T = 0.8, 0.5, 0.3, 1.7, 7
        y = [0.4, -0.7, 1.1, 0.2, -0.5, 0.9, 0.1]
        m_joint, v_joint = lgssm_smoother(y, a, q, r, s0)
        m_kf, v_kf, ll_kf = lgssm_kalman(y, a, q, r, s0)
        @test m_joint ≈ m_kf atol = 1e-12
        @test v_joint ≈ v_kf atol = 1e-12
        @test lgssm_loglik(y, a, q, r, s0) ≈ ll_kf atol = 1e-12

        P = [0.7 0.2 0.1; 0.15 0.7 0.15; 0.1 0.3 0.6]
        π0 = [0.5, 0.3, 0.2]
        loglik_obs = log.([0.3 0.5 0.2; 0.6 0.1 0.3; 0.2 0.2 0.6; 0.4 0.4 0.2; 0.1 0.8 0.1])
        post_fb, ll_fb = hmm_forward_backward(π0, P, loglik_obs)
        post_bf, ll_bf = hmm_brute_force(π0, P, loglik_obs)
        @test post_fb ≈ post_bf atol = 1e-12
        @test ll_fb ≈ ll_bf atol = 1e-12

        # A stationary π0 is a fixed point of the transition, which several tests rely on.
        π0_stat = stationary_distribution(P)
        @test transpose(P) * π0_stat ≈ π0_stat
    end
end

end
