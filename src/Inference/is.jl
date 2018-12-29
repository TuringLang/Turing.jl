init_samples(alg::IS; kwargs...) = Array{Sample}(undef, alg.n_particles)
function init_varinfo(model, spl::Sampler{<:IS}; stable = true, kwargs...)
    if stable
        return TypedVarInfo(default_varinfo(model, spl))
    else
        return nothing
    end
end

function _sample(args...; stable = true, kwargs...)
    if stable
        _sample_stable(args...)
    else
        _sample_unstable(args...)
    end
end

function _sample_stable(vi, samples, spl, model, alg::IS)
    n = spl.alg.n_particles
    for i = 1:n
        vi = empty!(deepcopy(vi))
        model(vi, spl)
        samples[i] = Sample(vi)
    end

    le = logsumexp(map(x->x[:lp], samples)) - log(n)

    Chain(exp(le), samples)
end

function _sample_unstable(vi, samples, spl, model, alg::IS)
    n = spl.alg.n_particles
    for i = 1:n
        vi = VarInfo()
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
