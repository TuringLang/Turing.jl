# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp

using Statistics: middle

"""
    gen_grad_func(vi::VarInfo, sampler::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, sampler, model)`.
"""
function gen_grad_func(vi::VarInfo, sampler::Sampler, model)
    function _f(θ::AbstractVector{<:Real})::Tuple{Float64,Vector{Float64}}
        return gradient_logp(θ, vi, model, sampler)
    end
    return _f
end

"""
    gen_lj_func(vi::VarInfo, sampler::Sampler, model)

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, sampler, model)`.
"""
function gen_lj_func(vi::VarInfo, sampler::Sampler, model)
    function _f(θ::AbstractVector{<:Real})::Float64
        vi[sampler] = θ
        logp = runmodel!(model, vi, sampler).logp
        return logp
    end
    return _f
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

function gen_metric(vi::VarInfo, spl::Sampler)
    return AdvancedHMC.UnitEuclideanMetric(length(vi[spl]))
end

function gen_metric(vi::VarInfo, spl::Sampler, ::UnitPreConditioner)
    return AdvancedHMC.UnitEuclideanMetric(length(vi[spl]))
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::DiagPreConditioner)
    return AdvancedHMC.DiagEuclideanMetric(pc.std.^2)
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::DensePreConditioner)
    return AdvancedHMC.DenseEuclideanMetric(pc.covar)
end

function _hmc_step(θ::AbstractVector{<:Real},
                   lj::Real,
                   logπ::Function,
                   _∂logπ∂θ::Function,
                   ϵ::Real,
                   λ::Real,
                   metric;
                   rev_func=nothing,
                   log_func=nothing,
                   )
    τ = max(1, round(Int, λ / ϵ))
    θ = Vector{Float64}(θ)

    h = AdvancedHMC.Hamiltonian(metric, logπ, x->_∂logπ∂θ(x)[2])
    prop = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(ϵ), τ)

    r = AdvancedHMC.rand_momentum(h)

    # I have to copy https://github.com/TuringLang/AdvancedHMC.jl/blob/master/src/trajectory.jl#L24
    # as the current interface doesn't return is_accept
    H = AdvancedHMC.hamiltonian_energy(h, θ, r)
    θ_new, r_new, is_valid = AdvancedHMC.step(prop.integrator, h, θ, r, prop.n_steps)
    H_new = AdvancedHMC.hamiltonian_energy(h, θ_new, r_new)
    # Accept via MH criteria
    is_accept, α = AdvancedHMC.mh_accept(H, H_new)
    if is_accept
        θ, r = θ_new, -r_new
    end
    lj_new = logπ(θ_new)

    return θ, lj_new, is_accept, is_valid, α
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/base_hmc.hpp
function find_good_eps(model, spl::Sampler{T}, vi::VarInfo) where T
    logπ = gen_lj_func(vi, spl, model)

    # AHMC only takes the gradient
    # TODO: unify two functions below
    _∂logπ∂θ = gen_grad_func(vi, spl, model)
    ∂logπ∂θ = x -> _∂logπ∂θ(x)[2]
    θ, lj = Vector{Float64}(vi[spl]), vi.logp

    # NOTE: currently force to use `UnitEuclideanMetric` - should be fine as
    #       this function shall be called before any sampling
    metric = AdvancedHMC.UnitEuclideanMetric(length(vi[spl]))
    h = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    init_eps = AdvancedHMC.find_good_eps(h, θ)

    vi[spl] = θ
    setlogp!(vi, lj)
    @info "[Turing] found initial ϵ: $init_eps"
    return init_eps
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
