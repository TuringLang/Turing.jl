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

    # rescaling due to loglikelihood weight and samples used
    c = weight / num_samples

    # If f: supp(p(z | x)) → ℝ then
    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(z̃)) + logabsdet(J(f⁻¹(z̃)))] + ℍ(q̃(z̃))
    #      = 𝔼[og p(x, z) - logabsdetjac(J(f(z)))] + ℍ(q̃(z̃))
    _, z, logjac, _ = forward(q)
    res = (logdensity(model, varinfo, z) - logjac) * c

    res += entropy(q)
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(q)
        res += (logdensity(model, varinfo, z) - logjac) * c
    end

    return res
end

function (elbo::ELBO)(
    alg::VariationalInference,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model,
    num_samples
)
    # extract the mean-field Gaussian params
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    return elbo(alg, q, model, θ, num_samples)
end

