using StatsFuns
using DistributionsAD
using Bijectors
using Bijectors: TransformedDistribution
using Random: AbstractRNG, GLOBAL_RNG
import Bijectors: bijector

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


# Overloading stuff from `AdvancedVI` to specialize for Turing
AdvancedVI.update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
AdvancedVI.update(td::TransformedDistribution, θ...) = transformed(AdvancedVI.update(td.dist, θ...), td.transform)
function AdvancedVI.update(td::TransformedDistribution{<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, ω = θ[1:length(td)], θ[length(td) + 1:end]
    return AdvancedVI.update(td, μ, softplus.(ω))
end

function AdvancedVI.vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    q = meanfield(model)
    return AdvancedVI.vi(model, alg, q; optimizer = optimizer)
end


function AdvancedVI.vi(model::Model, alg::ADVI, q::TransformedDistribution{<:TuringDiagMvNormal}; optimizer = TruncatedADAGrad())
    @debug "Optimizing ADVI..."
    # Initial parameters for mean-field approx
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    # Optimize
    AdvancedVI.optimize!(elbo, alg, q, make_logjoint(model), θ; optimizer = optimizer)

    # Return updated `Distribution`
    return AdvancedVI.update(q, θ)
end
