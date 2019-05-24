using ForwardDiff

import Distributions: _rand!


function jac_inv_transform(dist::Distribution, x::T where T<:Real)
    ForwardDiff.derivative(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::Array{T} where T <: Real)
    ForwardDiff.jacobian(x -> invlink(dist, x), x)
end

function center_diag_gaussian(x, μ, σ)
    # instead of creating a diagonal matrix, we just do elementwise multiplication
    (σ .^(-1)) .* (x - μ)
end

function center_diag_gaussian_inv(η, μ, σ)
    (η .* σ) + μ
end

struct MeanField{T, TDists <: AbstractVector{<: Distribution}} <: VariationalPosterior where T <: Real
    μ::Vector{T}
    ω::Vector{T}
    dists::TDists
    ranges::Vector{UnitRange{Int}}
end

# TODO: implement this following `Distribution` interface
Base.length(advi::MeanField) = length(advi.μ)

_rand!(rng::AbstractRNG, q::MeanField{T, TDists}, x::AbstractVector{T}) where {T<:Real, TDists <: AbstractVector{<: Distribution}} = begin
    # extract parameters for convenience
    μ, ω = q.μ, q.ω
    num_params = length(μ)

    for i = 1:size(q.dists, 1)
        prior = q.dists[i]
        r = q.ranges[i]

        # initials
        μ_i = μ[r]
        ω_i = ω[r]

        # # sample from VI posterior
        θ_acc = zeros(length(μ_i))

        η = randn(rng, length(μ_i))
        ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
        θ = invlink(prior, ζ)

        x[r] = θ
    end

    return x
end

"""
    ADVI(samplers_per_step = 10, max_iters = 5000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI <: VariationalInference
    samples_per_step
    max_iters
end

ADVI() = ADVI(10, 5000)

alg_str(::ADVI) = "ADVI"

vi(model::Model, alg::ADVI) = begin
    # setup
    var_info = Turing.VarInfo()
    model(var_info, Turing.SampleFromUniform())
    num_params = size(var_info.vals, 1)

    dists = var_info.dists
    ranges = var_info.ranges
    
    q = MeanField(zeros(num_params), zeros(num_params), dists, ranges)

    # construct objective
    elbo = ELBO()

    Turing.DEBUG && @debug "Optimizing ADVI..."
    μ, ω = optimize(elbo, alg, q, model)

    # TODO: make mutable instead?
    MeanField(μ, ω, dists, ranges) 
end

# TODO: implement optimize like this?
# (advi::ADVI)(elbo::EBLO, q::MeanField, model::Model) = begin
# end

function optimize(elbo::ELBO, alg::ADVI, q::MeanField, model::Model)
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters

    # number of previous gradients to use to compute `s` in adaGrad
    stepsize_num_prev = 10
    
    # setup
    var_info = Turing.VarInfo()
    model(var_info, Turing.SampleFromUniform())
    num_params = size(var_info.vals, 1)

    function f(x)
        # extract the mean-field Gaussian params
        μ, ω = x[1:num_params], x[num_params + 1: end]
        
        - elbo(q, model, μ, ω, samples_per_step)
    end

    # for every param we need a mean μ and variance ω
    x = zeros(2 * num_params)
    diff_result = DiffResults.GradientResult(x)

    # used for truncated adaGrad as suggested in (Blei et al, 2015). 
    η = 0.1
    τ = 1.0
    ρ = zeros(2 * num_params)
    s = zeros(2 * num_params)
    g² = zeros(2 * num_params)

    i = 0
    while (i < max_iters) # & converged # <= add criterion? A running mean maybe?
        # compute gradient
        ForwardDiff.gradient!(diff_result, f, x)
        
        # recursive implementation of updating the step-size
        # if beyound first sequence of steps we subtract of the previous g² before adding the next
        if i > stepsize_num_prev
            s -= g²
        end

        # update parameters for adaGrad
        g² .= DiffResults.gradient(diff_result).^2
        s += g²
        
        # compute stepsize
        @. ρ = η / (τ + sqrt(s))
        
        x .= x - ρ .* DiffResults.gradient(diff_result)
        Turing.DEBUG && @debug "Step $i" ρ DiffResults.value(diff_result) norm(DiffResults.gradient(diff_result))

        i += 1
    end

    μ, ω = x[1:num_params], x[num_params + 1: end]

    return μ, ω
end

function (elbo::ELBO)(q::MeanField, model::Model, μ::Vector{T}, ω::Vector{T}, num_samples) where T <: Real
# function objective(::ELBO, vi::ADVI, model::Model, μ::Vector{T}, ω::Vector{T}, num_samples) where T <: Real
    # ELBO
    
    # setup
    var_info = Turing.VarInfo()

    # initial `Var_Info` object
    model(var_info, Turing.SampleFromUniform())

    num_params = size(var_info.vals, 1)
    
    elbo_acc = 0.0

    for i = 1:num_samples
        # iterate through priors, sample and update
        for i = 1:size(var_info.dists, 1)
            prior = var_info.dists[i]
            r = var_info.ranges[i]

            # mean-field params for this set of model params
            μ_i = μ[r]
            ω_i = ω[r]

            # obtain samples from mean-field posterior approximation
            η = randn(length(μ_i))
            ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
            
            # inverse-transform back to original param space
            θ = invlink(prior, ζ)

            # update
            var_info.vals[r] = θ

            # add the log-det-jacobian of inverse transform
            elbo_acc += log(abs(det(jac_inv_transform(prior, ζ)))) / num_samples
        end

        # sample with updated variables
        model(var_info)
        elbo_acc += var_info.logp / num_samples
    end

    # add the term for the entropy of the variational posterior
    variational_posterior_entropy = sum(ω)
    elbo_acc += variational_posterior_entropy

    elbo_acc
end

function (elbo::ELBO)(q::MeanField, model::Model, num_samples)
# function objective(vi::ADVI, model::Model, num_samples)
    # extract the mean-field Gaussian params
    μ, ω = q.μ, q.ω

    elbo(vi, model, μ, ω, num_samples)
end
