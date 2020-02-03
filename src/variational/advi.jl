using StatsFuns
using DistributionsAD
using Bijectors
using Bijectors: TransformedDistribution
using Random: AbstractRNG, GLOBAL_RNG

update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)

# TODO: add these to DistributionsAD.jl and remove from here
Distributions.params(d::TuringDiagMvNormal) = (d.m, d.σ)

import StatsBase: entropy
function entropy(d::TuringDiagMvNormal)
    T = eltype(d.σ)
    return (DistributionsAD.length(d) * (T(log2π) + one(T)) / 2 + sum(log.(d.σ)))
end

import Bijectors: bijector
function bijector(model::Model; sym_to_ranges::Val{sym2ranges} = Val(false)) where {sym2ranges}
    varinfo = Turing.VarInfo(model)
    num_params = sum([size(varinfo.metadata[sym].vals, 1)
                      for sym ∈ keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym ∈ keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges)
                      for sym ∈ keys(varinfo.metadata)])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1

    # ranges might be discontinuous => values are vectors of ranges rather than just ranges
    sym_lookup = Dict{Symbol, Vector{UnitRange{Int}}}()
    for sym ∈ keys(varinfo.metadata)
        sym_lookup[sym] = Vector{UnitRange{Int}}()
        for r ∈ varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            push!(sym_lookup[sym], ranges[range_idx])
            range_idx += 1
        end

        idx += varinfo.metadata[sym].ranges[end][end]
    end

    bs = inv.(bijector.(tuple(dists...)))

    if sym2ranges
        return Stacked(bs, ranges), (; collect(zip(keys(sym_lookup), values(sym_lookup)))...)
    else
        return Stacked(bs, ranges)
    end
end

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
    d = TuringDiagMvNormal(μ, σ)
    bs = inv.(bijector.(tuple(dists...)))
    b = Stacked(bs, ranges)

    return transformed(d, b)
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


function vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    q = meanfield(model)
    return vi(model, alg, q; optimizer = optimizer)
end

# TODO: make more flexible, allowing other types of `q`
function vi(model, alg::ADVI, q::TransformedDistribution{<:TuringDiagMvNormal}; optimizer = TruncatedADAGrad())
    Turing.DEBUG && @debug "Optimizing ADVI..."
    # Initial parameters for mean-field approx
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    # Optimize
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # Return updated `Distribution`
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]
    return update(q, μ, softplus.(ω))
end

function vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa TransformedDistribution{<:TuringDiagMvNormal}
        μ, ω = θ[1:length(q)], θ[length(q) + 1:end]
        return update(q, μ, softplus.(ω))
    else
        # Otherwise we assume it's a mapping θ → q
        return q(θ)
    end
end


function optimize(elbo::ELBO, alg::ADVI, q, model, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)
    
    if model isa Model
        optimize!(elbo, alg, q, make_logjoint(model), θ; optimizer = optimizer)
    else
        # `model` assumed to be callable z ↦ p(x, z)
        optimize!(elbo, alg, q, model, θ; optimizer = optimizer)
    end

    return θ
end

"""
    make_logjoint(model; weight = 1.0)

Constructs the logjoint as a function of latent variables, i.e. the map z → p(x ∣ z) p(z).

The weight used to scale the likelihood, e.g. when doing stochastic gradient descent one needs to
use `DynamicPPL.MiniBatch` context to run the `Model` with a weight `num_total_obs / batch_size`.
"""
function make_logjoint(model; weight = 1.0)
    # setup
    ctx = DynamicPPL.MiniBatchContext(
        DynamicPPL.DefaultContext(),
        weight
    )
    varinfo = Turing.VarInfo(model, ctx)

    function logπ(z)
        varinfo = VarInfo(varinfo, SampleFromUniform(), z)
        model(varinfo)
        
        return varinfo.logp
    end

    return logπ
end

function logjoint(model, varinfo, z)
    varinfo = VarInfo(varinfo, SampleFromUniform(), z)
    model(varinfo)

    return varinfo.logp
end

function (elbo::ELBO)(alg::ADVI, q, logπ, θ, num_samples; kwargs...)
    return elbo(GLOBAL_RNG, alg, q, logπ, θ, num_samples; kwargs...)
end


function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADVI,
    q,
    model::Model,
    θ::AbstractVector{<:Real},
    num_samples;
    weight = 1.0,
    kwargs...
)   
    return elbo(rng, alg, q, make_logjoint(model; weight = weight), θ, num_samples; kwargs...)
end

function (elbo::ELBO)(
    alg::ADVI,
    q::TransformedDistribution{<:TuringDiagMvNormal},
    model::Model,
    num_samples;
    kwargs...
)
    # extract the mean-field Gaussian params
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    return elbo(alg, q, model, θ, num_samples; kwargs...)
end


function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADVI,
    q::TransformedDistribution{<:TuringDiagMvNormal},
    logπ::Function,
    θ::AbstractVector{<:Real},
    num_samples
)
    num_params = length(q)
    μ = θ[1:num_params]
    ω = θ[num_params + 1: end]

    # update the variational posterior
    q = update(q, μ, softplus.(ω))

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
    _, z, logjac, _ = forward(rng, q)
    res = (logπ(z) + logjac) / num_samples

    res += entropy(q.dist)
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(rng, q)
        res += (logπ(z) + logjac) / num_samples
    end

    return res
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADVI,
    getq::Function,
    logπ::Function,
    θ::AbstractVector{<:Real},
    num_samples
)
    # Update the variational posterior
    q = getq(θ)

    # ELBO computation
    _, z, logjac, _ = forward(rng, q)
    res = (logπ(z) + logjac) / num_samples

    res += entropy(q.dist)
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(rng, q)
        res += (logπ(z) + logjac) / num_samples
    end

    return res
end

# function (elbo::ELBO)(
#     rng::AbstractRNG,
#     alg::ADVI,
#     getq::Function,
#     logπ::Function,
#     θ::AbstractVector{<:Real},
#     estimator::AbstractEstimator;
#     weight = 1.0
# )
