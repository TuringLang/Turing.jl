using StatsFuns
using Turing.Core: update
using Bijectors

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

function _logdensity(model, varinfo, z)
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
    weight_ll = 1.0
) where T <: Real
    # setup
    varinfo = Turing.VarInfo(model)

    # extract params
    num_params = length(q)
    μ = θ[1:num_params]
    ω = θ[num_params + 1: end]

    # update the variational posterior
    q = update(q, μ, softplus.(ω))
    
    # sample from variational posterior
    # TODO: when batch computation is supported by Bijectors.jl use `forward` instead.
    samples = Distributions.rand(q, num_samples)

    # rescaling due to loglikelihood weight and samples used
    c = weight_ll / num_samples

    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(y)) + logabsdet(J(f⁻¹(y)))] + H(q(z))
    z = samples[:, 1]
    res = (_logdensity(model, varinfo, z) + logabsdetjacinv(q, z)) * c
    for i = 2:num_samples
        z = samples[:, i]
        res += (_logdensity(model, varinfo, z) + logabsdetjacinv(q, z)) * c
    end

    res += entropy(q)

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
