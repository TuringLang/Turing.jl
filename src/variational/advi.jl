using StatsFuns
using DistributionsAD
using Bijectors

update(d::TuringDiagNormal, μ, σ) = TuringDiagNormal(μ, σ)
update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)

# TODO: add these to DistributionsAD.jl and remove from here
Distributions.params(d::TuringDiagNormal) = (d.m, d.σ)
Distributions.length(d::TuringDiagNormal) = length(d.m)

import StatsBase: entropy
function entropy(d::TuringDiagNormal)
    T = eltype(d.σ)
    return (length(d) * (T(log2π) + one(T)) / 2 + sum(log.(d.σ)))
end

"""
    ADVI(samples_per_step = 1, max_iters = 1000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI{AD} <: VariationalInference{AD}
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI(args...) = ADVI{ADBackend()}(args...)
ADVI() = ADVI(1, 1000)

alg_str(::ADVI) = "ADVI"

"""
    meanfield(model::Model)

Creates a mean-field approximation with multivariate normal as underlying distribution.
"""
function meanfield(model::Model)
    # setup
    varinfo = Turing.VarInfo(model)
    num_params = sum([size(varinfo.metadata[sym].vals, 1)
                      for sym ∈ keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym ∈ keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges)
                      for sym ∈ keys(varinfo.metadata)])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1
    for sym ∈ keys(varinfo.metadata)
        for r ∈ varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            range_idx += 1
        end
        
        # append!(ranges, [idx .+ r for r ∈ varinfo.metadata[sym].ranges])
        idx += varinfo.metadata[sym].ranges[end][end]
    end

    # initial params
    μ = randn(num_params)
    σ = softplus.(randn(num_params))

    # construct variational posterior
    d = TuringDiagNormal(μ, σ)
    bs = inv.(bijector.(tuple(dists...)))
    b = Stacked(bs, ranges)

    return transformed(d, b)
end


function vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    q = meanfield(model)
    return vi(model, alg, q; optimizer = optimizer)
end

# TODO: make more flexible, allowing other types of `q`
function vi(
    model::Model,
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal};
    optimizer = TruncatedADAGrad()
)
    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]

    return update(q, μ, softplus.(ω))
end

function optimize(
    elbo::ELBO,
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model;
    optimizer = TruncatedADAGrad()
)
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function logdensity(model, varinfo, z)
    varinfo = VarInfo(varinfo, SampleFromUniform(), z)
    model(varinfo)

    return varinfo.logp
end

function (elbo::ELBO)(
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model,
    θ::AbstractVector{T},
    num_samples,
    weight = 1.0
) where T <: Real
    # setup
    varinfo = Turing.VarInfo(model)

    # extract params
    num_params = length(q)
    μ = θ[1:num_params]
    ω = θ[num_params + 1: end]

    # update the variational posterior
    q = update(q, μ, softplus.(ω))

    # rescaling due to loglikelihood weight and samples used
    c = weight / num_samples

    #   𝔼_q(z)[log p(xᵢ, z)]
    # = ∫ log p(xᵢ, z) q(z) dz
    # = ∫ log p(xᵢ, f(ϕ)) q(f(ϕ)) |det J_f(ϕ)| dϕ   (since change of variables)
    # = ∫ log p(xᵢ, f(ϕ)) q̃(ϕ) dϕ                   (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ)[log p(xᵢ, z)]

    #   𝔼_q(z)[log q(z)]
    # = ∫ q(f(ϕ)) log (q(f(ϕ))) |det J_f(ϕ)| dϕ     (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ) [log q(f(ϕ))]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ) - log |det J_f(ϕ)|]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ)] - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]
    # = - ℍ(q̃(ϕ)) - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]

    # Finally, the ELBO is given by
    # ELBO = 𝔼_q(z)[log p(xᵢ, z)] - 𝔼_q(z)[log q(z)]
    #      = 𝔼_q̃(ϕ)[log p(xᵢ, z)] + 𝔼_q̃(ϕ) [log |det J_f(ϕ)|] + ℍ(q̃(ϕ))

    # If f: supp(p(z | x)) → ℝ then
    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(z̃)) + logabsdet(J(f⁻¹(z̃)))] + ℍ(q̃(z̃))
    #      = 𝔼[log p(x, z) - logabsdetjac(J(f(z)))] + ℍ(q̃(z̃))

    # But our `forward(q)` is using f⁻¹: ℝ → supp(p(z | x)) going forward → `+ logjac`
    _, z, logjac, _ = forward(q)
    res = (logdensity(model, varinfo, z) + logjac) * c

    res += entropy(q.dist)
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(q)
        res += (logdensity(model, varinfo, z) + logjac) * c
    end

    return res
end

function (elbo::ELBO)(
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model,
    num_samples
)
    # extract the mean-field Gaussian params
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    return elbo(alg, q, model, θ, num_samples)
end

