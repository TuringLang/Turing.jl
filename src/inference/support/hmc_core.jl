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

function gen_metric(vi::VarInfo, spl::Sampler, ::AHMC.UnitPreConditioner)
    return AHMC.UnitEuclideanMetric(length(vi[spl]))
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::AHMC.DiagPreConditioner)
    return AHMC.DiagEuclideanMetric(AHMC.getM⁻¹(pc))
end

function gen_metric(vi::VarInfo, spl::Sampler, pc::AHMC.DensePreConditioner)
    return AHMC.DenseEuclideanMetric(AHMC.getM⁻¹(pc))
end

function _hmc_step(θ::AbstractVector{<:Real},
                   lj::Real,
                   logπ::Function,
                   ∂logπ∂θ::Function,
                   ϵ::Real,
                   τ::Real,
                   metric)
    θ = Vector{Float64}(θ)

    h = AHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
    prop = AHMC.StaticTrajectory(AHMC.Leapfrog(ϵ), τ)

    r = AHMC.rand_momentum(h)
    H = AHMC.hamiltonian_energy(h, θ, r)

    θ_new, r_new, α, H_new = AHMC.transition(prop, h, Vector{Float64}(θ), r)
    # NOTE: as `transition` doesn't return `is_accept`, I use `H == H_new` as a check
    is_accept = H != H_new  # If the new Hamiltonian enerygy is different
                            # from the old one, the sample was accepted.
    lj_new = logπ(θ_new)

    return θ_new, lj_new, is_accept, α
end
