using ForwardDiff
using Flux.Optimise

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

# Mean-field approximation used by ADVI
struct MeanField{T, TDists <: AbstractVector{<: Distribution}} <: VariationalPosterior where T <: Real
    μ::Vector{T}
    ω::Vector{T}
    dists::TDists
    ranges::Vector{UnitRange{Int}}
end

Base.length(advi::MeanField) = length(advi.μ)

_rand!(rng::AbstractRNG, q::MeanField{T, TDists}, x::AbstractVector{T}) where {T<:Real, TDists <: AbstractVector{<: Distribution}} = begin
    # extract parameters for convenience
    μ, ω = q.μ, q.ω
    num_params = length(q)

    for i = 1:size(q.dists, 1)
        prior = q.dists[i]
        r = q.ranges[i]

        # initials
        μ_i = μ[r]
        ω_i = ω[r]

        # # sample from VI posterior
        η = randn(rng, length(μ_i))
        ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
        θ = invlink(prior, ζ)

        x[r] = θ
    end

    return x
end

"""
    ADVI(samplers_per_step = 10, max_iters = 5000, opt = ADAGrad())

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI <: VariationalInference
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI() = ADVI(10, 5000)

alg_str(::ADVI) = "ADVI"

vi(model::Model, alg::ADVI; optimizer = ADAGrad()) = begin
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
    μ, ω = optimize(elbo, alg, q, model; optimizer = optimizer)

    # TODO: make mutable instead?
    MeanField(μ, ω, dists, ranges) 
end

# TODO: implement optimize like this?
# (advi::ADVI)(elbo::EBLO, q::MeanField, model::Model) = begin
# end

function optimize(elbo::ELBO, alg::ADVI, q::MeanField, model::Model; optimizer = ADAGrad())
    alg_name = alg_str(alg)
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

    # buffer
    x = zeros(2 * num_params)

    # HACK: re-use previous gradient `acc` if equal in value
    # Can cause issues if two entries have idenitical values
    vs = [v for v ∈ keys(optimizer.acc)]
    idx = findfirst(w -> vcat(q.μ, q.ω) == w, vs)
    if idx != nothing
        @info "[$alg_name] Re-using previous optimizer accumulator"
        x = vs[idx]
    end
    
    diff_result = DiffResults.GradientResult(x)

    # TODO: in (Blei et al, 2015) TRUNCATED ADAGrad is suggested; this is not available in Flux.Optimise
    # Maybe consider contributed a truncated ADAGrad to Flux.Optimise

    i = 0
    prog = PROGRESS[] ? ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0) : 0

    time_elapsed = @elapsed while (i < max_iters) # & converged # <= add criterion? A running mean maybe?
        # TODO: separate into a `grad(...)` call; need to manually provide `diff_result` buffers
        ForwardDiff.gradient!(diff_result, f, x)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = Optimise.apply!(optimizer, x, Δ)
        @. x = x - Δ
        
        Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result) norm(DiffResults.gradient(diff_result))
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    @info time_elapsed

    μ, ω = x[1:num_params], x[num_params + 1: end]

    return μ, ω
end

function (elbo::ELBO)(q::MeanField, model::Model, μ::Vector{T}, ω::Vector{T}, num_samples) where T <: Real
    # setup
    var_info = Turing.VarInfo()

    # initialize `VarInfo` object
    model(var_info, Turing.SampleFromUniform())

    num_params = length(q)
    
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

            # add the log-det-jacobian of inverse transform
            elbo_acc += log(abs(det(jac_inv_transform(prior, ζ)))) / num_samples
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

function (elbo::ELBO)(q::MeanField, model::Model, num_samples)
    # extract the mean-field Gaussian params
    μ, ω = q.μ, q.ω

    elbo(q, model, μ, ω, num_samples)
end
