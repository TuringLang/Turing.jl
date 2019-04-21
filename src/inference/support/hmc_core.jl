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
    return spl.alg.metricT(length(vi[spl]))
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
    H = AdvancedHMC.hamiltonian_energy(h, θ, r)

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

    return θ, lj_new, is_accept, α
end

# TODO: figure out why below doesn't work
# function _hmc_step(θ::AbstractVector{<:Real},
#                    lj::Real,
#                    logπ::Function,
#                    ∂logπ∂θ::Function,
#                    ϵ::Real,
#                    λ::Real,
#                    metric)
#     τ = max(1, round(Int, λ / ϵ))
#     θ = Vector{Float64}(θ)
#
#     h = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
#     prop = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(ϵ), τ)
#
#     r = AdvancedHMC.rand_momentum(h)
#     H = AdvancedHMC.hamiltonian_energy(h, θ, r)
#
#     θ_new, r_new, α, H_new = AdvancedHMC.transition(prop, h, Vector{Float64}(θ), r)
#     # NOTE: as `transition` doesn't return `is_accept`, I use `H == H_new` as a check
#     is_accept = H == H_new
#     lj_new = logπ(θ_new)
#
#     return θ_new, lj_new, is_accept, α
# end
