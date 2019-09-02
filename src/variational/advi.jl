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

function logdensity(model, varinfo, z)
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
    weight = 1.0
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
    c = weight / num_samples

    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(y)) + logabsdet(J(f⁻¹(y)))] + H(q(z))
    z = samples[:, 1]
    res = (logdensity(model, varinfo, z) + logabsdetjacinv(q, z)) * c

    res += entropy(q)
    
    for i = 2:num_samples
        z = samples[:, i]
        res += (logdensity(model, varinfo, z) + logabsdetjacinv(q, z)) * c
    end

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

