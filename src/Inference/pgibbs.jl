step(model, spl::Sampler{<:PG}, vi::AbstractVarInfo, _) = step(model, spl, vi)

function step(model, spl::Sampler{<:PG}, vi::AbstractVarInfo)
    particles = ParticleContainer{Trace}(model)

    vi.num_produce = 0;  # Reset num_produce before new sweep\.
    ref_particle = isempty(vi) ?
                  nothing :
                  forkr(Trace(model, spl, vi))

    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    if ref_particle == nothing
        push!(particles, spl.alg.n_particles, spl, vi)
    else
        push!(particles, spl.alg.n_particles-1, spl, vi)
        push!(particles, ref_particle)
    end

    while consume(particles) != Val{:done}
        resample!(particles, spl.alg.resampler, ref_particle)
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.info[:logevidence], particles.logE)

    return particles[indx].vi, true
end

init_samples(alg::PG, kwargs...) = Vector{Sample}()
function init_varinfo(model, spl::Sampler{<:PG}; resume_from = nothing, kwargs...)
    if resume_from == nothing
        return VarInfo()
    else
        return resume_from.info[:vi]
    end
end

function _sample(vi, samples, spl, model, alg::PG;
                  save_state=false,         # flag for state saving
                  resume_from=nothing,      # chain to continue
                  reuse_spl_n=0             # flag for spl re-using
                )

    ## custom resampling function for pgibbs
    ## re-inserts reteined particle after each resampling step
    time_total = zero(Float64)
    pm = nothing
    n = reuse_spl_n > 0 ?
        reuse_spl_n :
        alg.n_iters
    if PROGRESS[]
        spl.info[:progress] = ProgressMeter.Progress(n, 1, "[PG] Sampling...", 0)
    end

    for i = 1:n
        time_elapsed = @elapsed vi, _ = step(model, spl, vi)
        push!(samples, Sample(vi))
        samples[i].value[:elapsed] = time_elapsed

        time_total += time_elapsed

        if PROGRESS[]  && spl.alg.gid == 0
            ProgressMeter.next!(spl.info[:progress])
        end
    end

    @info("[PG] Finished with")
    @info("  Running time    = $time_total;")

    loge = exp.(mean(spl.info[:logevidence]))
    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
        pre_loge = resume_from.weight
        # Calculate new log-evidence
        pre_n = length(resume_from.value2)
        loge = exp.((log(pre_loge) * pre_n + log(loge) * n) / (pre_n + n))
    end
    c = Chain(loge, samples)       # wrap the result by Chain

    if save_state               # save state
        save!(c, spl, model, vi)
    end

    c
end

function assume(spl::Sampler{T}, dist::Distribution, vn::VarName, _::AbstractVarInfo) where T<:Union{PG,SMC}
    vi = current_trace().vi
    if isempty(getspace(spl)) || vn.sym in getspace(spl)
        if ~haskey(vi, vn)
            r = rand(dist)
            push!(vi, vn, r, dist, spl.alg.gid)
            spl.info[:cache_updated] = CACHERESET # sanity flag mask for getidcs and getranges
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(dist)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.alg.gid, vn)
            setorder!(vi, vn, vi.num_produce)
        else
            updategid!(vi, vn, spl)
            r = vi[vn]
        end
    else # vn belongs to other sampler <=> conditionning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(dist)
            push!(vi, vn, r, dist, -1)
        end
        acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    end
    r, zero(Real)
end

function assume(
                spl::Sampler{A}, 
                dists::Vector{D}, 
                vn::VarName, 
                var::Any, 
                vi::AbstractVarInfo
              ) where {A <: Union{PG,SMC}, D <: Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing assume statement")
end

function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:Union{PG,SMC}
    produce(logpdf(dist, value))
    zero(Real)
end

function observe(spl::Sampler{A}, ds::Vector{D}, value::Any, vi::AbstractVarInfo) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing observe statement")
end
