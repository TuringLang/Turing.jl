"""
    bijector(model::Model[, sym2ranges = Val(false)])

Returns a `Stacked <: Bijector` which maps from the support of the posterior to ℝᵈ with `d`
denoting the dimensionality of the latent variables.
"""
function Bijectors.bijector(
    model::DynamicPPL.Model,
    ::Val{sym2ranges} = Val(false),
) where {sym2ranges}
    varinfo = DynamicPPL.VarInfo(model)
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

    bs = Bijectors.bijector.(tuple(dists...))
    rs = tuple(ranges...)

    if sym2ranges
        return (
            Bijectors.Stacked(bs, rs),
            (; collect(zip(keys(sym_lookup), values(sym_lookup)))...),
        )
    else
        return Bijectors.Stacked(bs, rs)
    end
end

"""
    meanfield([rng, ]model::Model)

Creates a mean-field approximation with multivariate normal as underlying distribution.
"""
meanfield(model::DynamicPPL.Model) = meanfield(Random.GLOBAL_RNG, model)
function meanfield(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    # setup
    varinfo = DynamicPPL.VarInfo(model)
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
    σ = StatsFuns.softplus.(randn(rng, num_params))

    # construct variational posterior
    d = DistributionsAD.TuringDiagMvNormal(μ, σ)
    bs = inv.(Bijectors.bijector.(tuple(dists...)))
    b = Bijectors.Stacked(bs, ranges)

    return Bijectors.transformed(d, b)
end

# Overloading stuff from `AdvancedVI` to specialize for Turing
function AdvancedVI.update(d::DistributionsAD.TuringDiagMvNormal, μ, σ)
    return DistributionsAD.TuringDiagMvNormal(μ, σ)
end
function AdvancedVI.update(td::Bijectors.TransformedDistribution, θ...)
    return Bijectors.transformed(AdvancedVI.update(td.dist, θ...), td.transform)
end
function AdvancedVI.update(
    td::Bijectors.TransformedDistribution{<:DistributionsAD.TuringDiagMvNormal},
    θ::AbstractArray,
)
    μ, ω = θ[1:length(td)], θ[length(td) + 1:end]
    return AdvancedVI.update(td, μ, StatsFuns.softplus.(ω))
end

function AdvancedVI.vi(
    model::DynamicPPL.Model,
    alg::AdvancedVI.ADVI;
    optimizer = AdvancedVI.TruncatedADAGrad(),
)
    q = meanfield(model)
    return AdvancedVI.vi(model, alg, q; optimizer = optimizer)
end


function AdvancedVI.vi(
    model::DynamicPPL.Model,
    alg::AdvancedVI.ADVI,
    q::Bijectors.TransformedDistribution{<:DistributionsAD.TuringDiagMvNormal};
    optimizer = AdvancedVI.TruncatedADAGrad(),
)
    # Initial parameters for mean-field approx
    μ, σs = StatsBase.params(q)
    θ = vcat(μ, StatsFuns.invsoftplus.(σs))

    # Optimize
    AdvancedVI.optimize!(elbo, alg, q, make_logjoint(model), θ; optimizer = optimizer)

    # Return updated `Distribution`
    return AdvancedVI.update(q, θ)
end
