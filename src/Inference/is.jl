init_samples(alg::IS; kwargs...) = Array{Sample}(undef, alg.n_particles)

function _sample(vi, samples, spl, model, alg::IS)
    n = spl.alg.n_particles
    for i = 1:n
        vi = empty!(deepcopy(vi))
        model(vi, spl)
        samples[i] = Sample(vi)
    end

    le = logsumexp(map(x->x[:lp], samples)) - log(n)

    Chain(exp(le), samples)
end

function assume(spl::Sampler{<:IS}, dist::Distribution, vn::VarName, vi::AbstractVarInfo)
    r = rand(dist)
    push!(vi, vn, r, dist, 0)
    r, zero(Real)
end

function observe(spl::Sampler{<:IS}, dist::Distribution, value::Any, vi::AbstractVarInfo)
    # acclogp!(vi, logpdf(dist, value))
    logpdf(dist, value)
end
