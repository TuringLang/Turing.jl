# TODO(torfjelde): Find a better solution.
struct Vec{N,B} <: Bijectors.Bijector
    b::B
    size::NTuple{N, Int}
end

Bijectors.inverse(f::Vec) = Vec(Bijectors.inverse(f.b), f.size)

Bijectors.output_length(f::Vec, sz) = Bijectors.output_length(f.b, sz)
Bijectors.output_length(f::Vec, n::Int) = Bijectors.output_length(f.b, n)

function Bijectors.with_logabsdet_jacobian(f::Vec, x)
    return Bijectors.transform(f, x), Bijectors.logabsdetjac(f, x)
end

function Bijectors.transform(f::Vec, x::AbstractVector)
    # Reshape into shape compatible with wrapped bijector and then `vec` again.
    return vec(f.b(reshape(x, f.size)))
end

function Bijectors.transform(f::Vec{N,<:Bijectors.Inverse}, x::AbstractVector) where N
    # Reshape into shape compatible with original (forward) bijector and then `vec` again.
    return vec(f.b(reshape(x, Bijectors.output_length(f.b.orig, prod(f.size)))))
end

function Bijectors.transform(f::Vec, x::AbstractMatrix)
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
        if d isa Distributions.UnivariateDistribution
            b
        else
            Vec(b, size(d))
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

Creates a mean-field approximation with a unit normal as underlying distribution.
"""
meanfield(model::DynamicPPL.Model) = meanfield(Random.default_rng(), model)
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
    return Bijectors.transformed(d, Bijectors.inverse(b))
end

ADVI() = AdvancedVI.ADVI(1)

function vi(
    model::DynamicPPL.Model,
    obj::ADVI,
    max_iters::Int;
    kwargs...
)
    q_trans = meanfield(model)
    return vi(model, obj, q_trans, max_iters; kwargs...)
end

function vi(
    model::DynamicPPL.Model,
    obj::ADVI,
    q,
    max_iters::Int;
    kwargs...
)
    varinfo = DynamicPPL.VarInfo(model)
    b = Bijectors.bijector(model)
    prob = DynamicPPL.LogDensityFunction(model, varinfo)

    q, _, _ = optimize(
        prob, obj, q, max_iters; adbackend=ADBackend(), kwargs...
    )
    return q
end
