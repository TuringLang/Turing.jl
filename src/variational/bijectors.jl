
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
    model::DynamicPPL.Model, ::Val{sym2ranges}=Val(false); varinfo=DynamicPPL.VarInfo(model)
) where {sym2ranges}
    num_params = sum([
        size(varinfo.metadata[sym].vals, 1) for sym in keys(varinfo.metadata)
    ])

    dists = vcat([varinfo.metadata[sym].dists for sym in keys(varinfo.metadata)]...)

    num_ranges = sum([
        length(varinfo.metadata[sym].ranges) for sym in keys(varinfo.metadata)
    ])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1

    # ranges might be discontinuous => values are vectors of ranges rather than just ranges
    sym_lookup = Dict{Symbol,Vector{UnitRange{Int}}}()
    for sym in keys(varinfo.metadata)
        sym_lookup[sym] = Vector{UnitRange{Int}}()
        for r in varinfo.metadata[sym].ranges
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
