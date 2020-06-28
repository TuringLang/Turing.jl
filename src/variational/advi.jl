using StatsFuns
using DistributionsAD
using Bijectors
using Bijectors: TransformedDistribution
using Random: AbstractRNG, GLOBAL_RNG
import Bijectors: bijector

update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)
function update(td::TransformedDistribution{<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, ω = θ[1:length(td)], θ[length(td) + 1:end]
    return update(td, μ, softplus.(ω))
end

"""
    bijector(model::Model; sym_to_ranges = Val(false))

Returns a `Stacked <: Bijector` which maps from the support of the posterior to ℝᵈ with `d`
denoting the dimensionality of the latent variables.
"""
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

    bs = bijector.(tuple(dists...))

    if sym2ranges
        return Stacked(bs, ranges), (; collect(zip(keys(sym_lookup), values(sym_lookup)))...)
    else
        return Stacked(bs, ranges)
    end
end

"""
    meanfield(model::Model)
    meanfield(rng::AbstractRNG, model::Model)

Creates a mean-field approximation with multivariate normal as underlying distribution.
"""
meanfield(model::Model) = meanfield(GLOBAL_RNG, model)
function meanfield(rng::AbstractRNG, model::Model)
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
    μ = randn(rng, num_params)
    σ = softplus.(randn(rng, num_params))

    # construct variational posterior
    d = TuringDiagMvNormal(μ, σ)
    bs = inv.(bijector.(tuple(dists...)))
    b = Stacked(bs, ranges)

    return transformed(d, b)
end

"""
$(TYPEDEF)

Automatic Differentiation Variational Inference (ADVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct ADVI{AD} <: VariationalInference{AD}
    "Number of samples used to estimate the ELBO in each optimization step."
    samples_per_step::Int
    "Maximum number of gradient steps."
    max_iters::Int
end

"""
    ADVI([samples_per_step=1, max_iters=1000])

Create an [`ADVI`](@ref) with the currently enabled automatic differentiation backend
`ADBackend()`.
"""
function ADVI(samples_per_step::Int=1, max_iters::Int=1000)
    return ADVI{ADBackend()}(samples_per_step, max_iters)
end

DynamicPPL.alg_str(::ADVI) = "ADVI"

function vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    q = meanfield(model)
    return vi(model, alg, q; optimizer = optimizer)
end

function vi(model, alg::ADVI, q::TransformedDistribution{<:TuringDiagMvNormal}; optimizer = TruncatedADAGrad())
    Turing.DEBUG && @debug "Optimizing ADVI..."
    # Initial parameters for mean-field approx
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    # Optimize
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # Return updated `Distribution`
    return update(q, θ)
end

function vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa Distribution
        return update(q, θ)
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

# WITHOUT updating parameters inside ELBO
function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADVI,
    q::VariationalPosterior,
    logπ::Function,
    num_samples
)
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

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(rng, q)
        res += (logπ(z) + logjac) / num_samples
    end

    return res
end

