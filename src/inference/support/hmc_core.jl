"""
    gen_grad_func(vi::VarInfo, spl::Sampler, model)

Generate a function that takes a vector of reals `θ` and compute the logpdf and
gradient at `θ` for the model specified by `(vi, spl, model)`.
"""
function gen_grad_func(vi::VarInfo, spl::Sampler, model)
    function ∂logπ∂θ(x)::Vector{Float64}
        x_old, lj_old = vi[spl], vi.logp
        _, deriv = gradient_logp(x, vi, model, spl)
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return deriv
    end
    return ∂logπ∂θ
end

"""
    gen_lj_func(vi::VarInfo, spl::Sampler, model)

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, spl, model)`.
"""
function gen_lj_func(vi::VarInfo, spl::Sampler, model)
    function logπ(x)::Float64
        x_old, lj_old = vi[spl], vi.logp
        vi[spl] = x
        runmodel!(model, vi, spl).logp
        lj = vi.logp
        vi[spl] = x_old
        setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

function gen_metric(vi::VarInfo, spl::Sampler)
    return AdvancedHMC.UnitEuclideanMetric(length(vi[spl]))
end

function gen_metric(vi::VarInfo, spl::Sampler, ::AdvancedHMC.UnitPreConditioner)
    return AdvancedHMC.UnitEuclideanMetric(length(vi[spl]))
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::AdvancedHMC.DiagPreConditioner)
    return AdvancedHMC.DiagEuclideanMetric(AdvancedHMC.getM⁻¹(pc))
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::AdvancedHMC.DensePreConditioner)
    return AdvancedHMC.DenseEuclideanMetric(AdvancedHMC.getM⁻¹(pc))
end

function _hmc_step(θ::AbstractVector{<:Real},
                   lj::Real,
                   logπ::Function,
                   ∂logπ∂θ::Function,
                   ϵ::Real,
                   λ::Real,
                   metric)
    τ = max(1, round(Int, λ / ϵ))
    θ = Vector{Float64}(θ)

    h = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
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
    ∂logπ∂θ = gen_grad_func(vi, spl, model)
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
