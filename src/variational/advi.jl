# TODO(torfjelde): Find a better solution.
struct Vec{N, B<:Bijectors.Bijector{N}} <: Bijectors.Bijector{1}
    b::B
    size::NTuple{N, Int}
end

Bijectors.inverse(f::Vec) = Vec(Bijectors.inverse(f.b), f.size)

function (f::Vec)(x::AbstractVector)
    # Reshape into shape compatible with wrapped bijector and then `vec` again.
    return vec(f.b(reshape(x, f.size)))
end

function (f::Vec)(x::AbstractMatrix)
    # At the moment we do batching for higher-than-1-dim spaces by simply using
    # lists of inputs rather than `AbstractArray` with `N + 1` dimension.
    cols = Iterators.Stateful(eachcol(x))
    # Make `init` a matrix to ensure type-stability
    init = reshape(f(first(cols)), :, 1)
    return mapreduce(f, hcat, cols; init = init)
end

function Bijectors.logabsdetjac(f::Vec, x::AbstractVector)
    return Bijectors.logabsdetjac(f.b, reshape(x, f.size))
end

function Bijectors.logabsdetjac(f::Vec, x::AbstractMatrix)
    return map(eachcol(x)) do x_
        Bijectors.logabsdetjac(f, x_)
    end
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

        return if Bijectors.dimension(b) > 1
            Vec(b, size(d))
        else
            b
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
meanfield(model::DynamicPPL.Model) = meanfield(Random.GLOBAL_RNG, model)
function meanfield(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    # Setup.
    varinfo = DynamicPPL.VarInfo(model)
    num_params = length(varinfo[DynamicPPL.SampleFromPrior()])

    # initial params
    μ = randn(rng, num_params)
    σ = StatsFuns.softplus.(randn(rng, num_params))

    # Construct the base family.
    d = DistributionsAD.TuringDiagMvNormal(μ, σ)

    # Construct the bijector constrained → unconstrained.
    b = Bijectors.bijector(model; varinfo=varinfo)

    # We want to transform from unconstrained space to constrained,
    # hence we need the inverse of `b`.
    return Bijectors.transformed(d, inv(b))
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
