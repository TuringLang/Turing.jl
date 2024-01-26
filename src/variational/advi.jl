# TODO: Move to Bijectors.jl if we find further use for this.
"""
    wrap_in_vec_reshape(f, in_size)

Wraps a bijector `f` such that it operates on vectors of length `prod(in_size)` and produces
a vector of length `prod(Bijectors.output(f, in_size))`.
"""
function wrap_in_vec_reshape(f, in_size)
    vec_in_length = prod(in_size)
    reshape_inner = Bijectors.Reshape((vec_in_length,), in_size)
    out_size = Bijectors.output_size(f, in_size)
    vec_out_length = prod(out_size)
    reshape_outer = Bijectors.Reshape(out_size, (vec_out_length,))
    return reshape_outer ∘ f ∘ reshape_inner
end


"""
    bijector(model::Model[, sym2ranges = Val(false)])

Returns a `Stacked <: Bijector` which maps from the support of the posterior to ℝᵈ with `d`
denoting the dimensionality of the latent variables.
"""
function Bijectors.bijector(
    model::DynamicPPL.Model,
    ::Val{sym2ranges} = Val(false);
    varinfo = DynamicPPL.VarInfo(model)
) where {sym2ranges}
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

    bs = map(tuple(dists...)) do d
        b = Bijectors.bijector(d)
        if d isa Distributions.UnivariateDistribution
            b
        else
            wrap_in_vec_reshape(b, size(d))
        end
    end

    if sym2ranges
        return (
            Bijectors.Stacked(bs, ranges),
            (; collect(zip(keys(sym_lookup), values(sym_lookup)))...),
        )
    else
        return Bijectors.Stacked(bs, ranges)
    end
end

"""
    meanfield([rng, ]model::Model)

Creates a mean-field approximation with multivariate normal as underlying distribution.
"""
meanfield(model::DynamicPPL.Model) = meanfield(Random.default_rng(), model)
function meanfield(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    # Setup.
    varinfo = DynamicPPL.VarInfo(model)
    # Use linked `varinfo` to determine the correct number of parameters.
    # TODO: Replace with `length` once this is implemented for `VarInfo`.
    varinfo_linked = DynamicPPL.link(varinfo, model)
    num_params = length(varinfo_linked[:])

    # initial params
    μ = randn(rng, num_params)
    σ = StatsFuns.softplus.(randn(rng, num_params))

    # Construct the base family.
    d = DistributionsAD.TuringDiagMvNormal(μ, σ)

    # Construct the bijector constrained → unconstrained.
    b = Bijectors.bijector(model; varinfo=varinfo)

    # We want to transform from unconstrained space to constrained,
    # hence we need the inverse of `b`.
    return Bijectors.transformed(d, Bijectors.inverse(b))
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
    # `length(td.dist) != length(td)` if `td.transform` changes the dimensionality,
    # so we need to use the length of the underlying distribution `td.dist` here.
    # TODO: Check if we can get away with `view` instead of `getindex` for all AD backends.
    μ, ω = θ[begin:(begin + length(td.dist) - 1)], θ[(begin + length(td.dist)):end]
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
