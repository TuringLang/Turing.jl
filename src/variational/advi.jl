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
    model::DynamicPPL.Model, alg::AdvancedVI.ADVI; optimizer=AdvancedVI.TruncatedADAGrad()
)
    q = meanfield(model)
    return AdvancedVI.vi(model, alg, q; optimizer=optimizer)
end

function AdvancedVI.vi(
    model::DynamicPPL.Model,
    alg::AdvancedVI.ADVI,
    q::Bijectors.TransformedDistribution{<:DistributionsAD.TuringDiagMvNormal};
    optimizer=AdvancedVI.TruncatedADAGrad(),
)
    # Initial parameters for mean-field approx
    μ, σs = StatsBase.params(q)
    θ = vcat(μ, StatsFuns.invsoftplus.(σs))

    # Optimize
    AdvancedVI.optimize!(elbo, alg, q, make_logjoint(model), θ; optimizer=optimizer)

    # Return updated `Distribution`
    return AdvancedVI.update(q, θ)
end
