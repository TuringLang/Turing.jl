using ForwardDiff


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


abstract type VariationalInference <: InferenceAlgorithm end

"""
    sample(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal to `num_samples`.
"""
function sample(vi::VariationalInference, num_samples) end

"""
    objective(vi::VariationalInference, num_samples)

Computes empirical estimates of ELBO for the given VI method using number of samples equal to `num_samples`.
"""
function objective(vi::VariationalInference, num_samples) end

"""
    optimize(vi::VariationalInference)

Finds parameters which maximizes the ELBO for the given VI method.
"""
function optimize(vi::VariationalInference) end

### ADVI ###

"""
    ADVI(model::Turing.Model)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
mutable struct ADVI{T} <: VariationalInference
    n_iters :: Int
    space :: Set{T}
    samples_per_step::Int
    max_iters::Int
end

# ADVI(n_vars, n_iters) = ADVI(zeros(n_vars), zeros(n_vars), n_iters, Set{Float64}())
ADVI(n_iters; samples_per_step=5, max_iters=5000) = ADVI(n_iters, Set{Float64}(), samples_per_step, max_iters)

function sample(model::Model, alg::ADVI, save_state=false, resume_from=nothing, reuse_spl_n=0)
    num_samples = alg.n_iters
    
    # setup
    spl = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        Sampler(alg, model)
    if resume_from != nothing
        spl.selector = resume_from.info[:spl].selector
    end
    
    var_info = if resume_from == nothing
        vi_ = VarInfo()
        model(vi_, SampleFromUniform())
        vi_
    else
        resume_from.info[:vi]
    end

    time_total = 0.0
    n = reuse_spl_n > 0 ?
        reuse_spl_n :
        num_samples

    if spl.selector.tag == :default
        runmodel!(model, var_info, spl)
    end

    # extract number of parameters for the model
    num_params = size(var_info.vals, 1)

    # optimize
    # TODO: optimization options needs to be part of the algorithm struct
    μ, ω = optimize(alg, model)
    # alg.μ, alg.ω = μ, ω

    # buffer
    samples = zeros(num_samples, num_params)

    # No point in looking at the progress here since it's so quick?
    alg_name = alg_str(alg)
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_name] Sampling...", 0))
    
    for i = 1:size(var_info.dists, 1)
        prior = var_info.dists[i]
        r = var_info.ranges[i]

        # initials
        μ_i = μ[r]
        ω_i = ω[r]

        # # sample from VI posterior
        θ_acc = zeros(length(μ_i))

        for j = 1:num_samples
            η = randn(length(μ_i))
            ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
            θ = invlink(prior, ζ)

            samples[j, r] = θ
        end
    end

    # # TODO: have to construct `Sample` objects....
    # for j = 1:num_samples
    #     # re-iterate through and turn array into samples
    #     for i, vn in enumerate(keys(vn))
            
    #     end
    # end

    # construct the sampler chain
    # names = keys(vi)
    # samples = Chain()

    return samples
end

function optimize(vi::ADVI, model::Model)
    samples_per_step = vi.samples_per_step
    max_iters = vi.max_iters
    
    # setup
    var_info = Turing.VarInfo()
    model(var_info, Turing.SampleFromUniform())
    num_params = size(var_info.vals, 1)

    function f(x)
        # extract the mean-field Gaussian params
        μ, ω = x[1:num_params], x[num_params + 1: end]
        
        - objective(vi, model, μ, ω, samples_per_step)
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

    # number of previous gradients to use to compute `s` in adaGrad
    stepsize_num_prev = 10

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

function objective(vi::ADVI, model::Model, μ::Vector{T}, ω::Vector{T}, num_samples) where T <: Real
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

function objective(vi::ADVI, model::Model, num_samples)
    # extract the mean-field Gaussian params
    μ, ω = vi.μ, vi.ω

    elbo(vi, model, μ, ω, num_samples)
end


# TODO: implement `assume` and `observe`; probably just default
alg_str(::ADVI) = "ADVI"

function Sampler(alg::ADVI, s::Selector)
    info = Dict{Symbol, Any}()
    return Sampler(alg, info, s)
end

function assume(spl::Sampler{<:ADVI}, dist::Distribution, vn::VarName, vi::VarInfo)
    Turing.DEBUG && @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "vn = $vn"
    Turing.DEBUG && @debug "r = $r" "typeof(r)=$(typeof(r))"

    r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

observe(spl::Sampler{<:ADVI},
    d::Distribution,
    value::Any,
    vi::VarInfo) = observe(nothing, d, value, vi)
