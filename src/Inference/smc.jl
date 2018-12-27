function step(model, spl::Sampler{<:SMC}, vi::AbstractVarInfo)
    particles = ParticleContainer{Trace}(model)
    vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    push!(particles, spl.alg.n_particles, spl, vi)

    while consume(particles) != Val{:done}
        ess = effectiveSampleSize(particles)
        if ess <= spl.alg.resampler_threshold * length(particles)
            resample!(particles,spl.alg.resampler)
        end
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.info[:logevidence], particles.logE)

    particles[indx].vi
end

init_samples(::SMC; kwargs...) = nothing
init_varinfo(::Any, ::Sampler{<:SMC}) = nothing

## wrapper for smc: run the sampler, collect results.
function _sample(_vi, _samples, spl, model, alg::SMC)
    particles = ParticleContainer{Trace}(model)
    push!(particles, spl.alg.n_particles, spl, VarInfo())

    while consume(particles) != Val{:done}
        ess = effectiveSampleSize(particles)
        if ess <= spl.alg.resampler_threshold * length(particles)
            resample!(particles,spl.alg.resampler)
        end
    end
    w, samples = getsample(particles)
    res = Chain(w, samples)
end
