using Random: GLOBAL_RNG

struct ELBO <: VariationalObjective end

function (elbo::ELBO)(alg, q, logπ, num_samples; kwargs...)
    return elbo(GLOBAL_RNG, alg, q, logπ, num_samples; kwargs...)
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::VariationalInference,
    q,
    model::Model,
    num_samples;
    weight = 1.0,
    kwargs...
)
    return elbo(rng, alg, q, make_logjoint(model; weight = weight), num_samples; kwargs...)
end

const elbo = ELBO()
