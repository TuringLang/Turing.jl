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

function vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    # setup
    varinfo = Turing.VarInfo(model)
    num_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym ∈ keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges) for sym ∈ keys(varinfo.metadata)])
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

    q = Variational.MeanField(zeros(num_params), zeros(num_params), dists, ranges)
    
    # construct objective
    elbo = ELBO()

    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]

    return MeanField(μ, ω, dists, ranges) 
end

# TODO: implement optimize like this?
# (advi::ADVI)(elbo::EBLO, q::MeanField, model::Model) = begin
# end

function optimize(elbo::ELBO, alg::ADVI, q::MeanField, model::Model; optimizer = TruncatedADAGrad())
    θ = randn(2 * length(q))
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function (elbo::ELBO)(
    alg::ADVI,
    q::MeanField,
    model::Model,
    θ::AbstractVector{<:Real},
    num_samples
)
    # setup
    varinfo = Turing.VarInfo(model)

    T = eltype(θ)
    num_params = length(q)
    μ, ω = θ[1:num_params], θ[num_params + 1: end]
    
    elbo_acc = 0.0

    # TODO: instead use `rand(q, num_samples)` and iterate through?
    # Requires new interface for Bijectors.jl

    for i = 1:num_samples
        # iterate through priors, sample and update
        idx = 0
        z = zeros(T, num_params)
        
        for sym ∈ keys(varinfo.metadata)
            md = varinfo.metadata[sym]
            
            for i = 1:size(md.dists, 1)
                prior = md.dists[i]
                r = md.ranges[i] .+ idx

                # mean-field params for this set of model params
                μ_i = μ[r]
                ω_i = ω[r]

                # obtain samples from mean-field posterior approximation
                η = randn(length(μ_i))
                ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
                
                # inverse-transform back to domain of original priro
                z[r] .= invlink(prior, ζ)

                # update
                # @info θ
                # z[md.ranges[i]] .= θ
                # @info md.vals

                # add the log-det-jacobian of inverse transform;
                # `logabsdet` returns `(log(abs(det(M))), sign(det(M)))` so return first entry
                # add `eps` to ensure SingularException does not occurr in `logabsdet`
                elbo_acc += logabsdet(jac_inv_transform(prior, ζ) .+ eps(T))[1] / num_samples
            end

            idx += md.ranges[end][end]
        end
        
        # compute log density
        varinfo = VarInfo(varinfo, SampleFromUniform(), z)
        model(varinfo)
        elbo_acc += getlogp(varinfo) / num_samples
    end

    # add the term for the entropy of the variational posterior
    variational_posterior_entropy = sum(ω)
    elbo_acc += variational_posterior_entropy

    return elbo_acc
end

function (elbo::ELBO)(alg::ADVI, q::MeanField, model::Model, num_samples)
    # extract the mean-field Gaussian params
    θ = vcat(q.μ, q.ω)

    return elbo(alg, q, model, θ, num_samples)
end
