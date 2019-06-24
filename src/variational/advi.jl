"""
    ADVI(samples_per_step = 10, max_iters = 5000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI{AD} <: VariationalInference{AD}
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI(args...) = ADVI{ADBackend()}(args...)
ADVI() = ADVI(10, 5000)

alg_str(::ADVI) = "ADVI"

function vi(model::Model, alg::ADVI; optimizer = ADAGrad())
    # setup
    var_info = VarInfo()
    model(var_info, SampleFromUniform())
    num_params = size(var_info.vals, 1)

    dists = var_info.dists
    ranges = var_info.ranges
    
    q = MeanField(zeros(num_params), zeros(num_params), dists, ranges)

    # construct objective
    elbo = ELBO()

    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]

    # TODO: make mutable instead?
    return MeanField(μ, ω, dists, ranges) 
end

# TODO: implement optimize like this?
# (advi::ADVI)(elbo::EBLO, q::MeanField, model::Model) = begin
# end

function optimize(elbo::ELBO, alg::ADVI, q::MeanField, model::Model; optimizer = ADAGrad())
    θ = randn(2 * length(q))
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function (elbo::ELBO)(
    alg::ADVI,
    q::MeanField,
    model::Model,
    θ::AbstractVector{T},
    num_samples
) where T <: Real
    # setup
    var_info = Turing.VarInfo()

    # initialize `VarInfo` object
    model(var_info, Turing.SampleFromUniform())

    num_params = length(q)
    μ, ω = θ[1:num_params], θ[num_params + 1: end]
    
    elbo_acc = 0.0

    # TODO: instead use `rand(q, num_samples)` and iterate through?

    for i = 1:num_samples
        # iterate through priors, sample and update
        for i = 1:size(q.dists, 1)
            prior = q.dists[i]
            r = q.ranges[i]

            # mean-field params for this set of model params
            μ_i = μ[r]
            ω_i = ω[r]

            # obtain samples from mean-field posterior approximation
            η = randn(length(μ_i))
            ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
            
            # inverse-transform back to domain of original priro
            θ = invlink(prior, ζ)

            # update
            var_info.vals[r] = θ

            # add the log-det-jacobian of inverse transform;
            # `logabsdet` returns `(log(abs(det(M))), sign(det(M)))` so return first entry
            elbo_acc += logabsdet(jac_inv_transform(prior, ζ))[1] / num_samples
        end

        # compute log density
        model(var_info)
        elbo_acc += var_info.logp / num_samples
    end

    # add the term for the entropy of the variational posterior
    variational_posterior_entropy = sum(ω)
    elbo_acc += variational_posterior_entropy

    elbo_acc
end

function (elbo::ELBO)(alg::ADVI, q::MeanField, model::Model, num_samples)
    # extract the mean-field Gaussian params
    θ = vcat(q.μ, q.ω)

    elbo(alg, q, model, θ, num_samples)
end
