"""
    PG(n_particles::Int, n_iters::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x[1] ~ Normal(m, sqrt(s))
    x[2] ~ Normal(m, sqrt(s))
    return s, m
end

sample(gdemo([1.5, 2]), PG(100, 100))
```
"""
mutable struct PG{space, F} <: AbstractGibbs{space}
    n_particles           ::    Int         # number of particles used
    n_iters               ::    Int         # number of iterations
    resampler             ::    F           # function to resample
    gid                   ::    Int         # group ID
end
PG(n1::Int, n2::Int) = PG{(), typeof(resample_systematic)}(n1, n2, resample_systematic, 0)
function PG(n1::Int, n2::Int, space...)
    F = typeof(resample_systematic)
    PG{space, F}(n1, n2, resample_systematic, 0)
end
function PG(alg::PG{space, F}, new_gid::Int) where {space, F}
    return PG{space, F}(alg.n_particles, alg.n_iters, alg.resampler, new_gid)
end
PG{space, F}(alg::PG, new_gid::Int) where {space, F} = PG{space, F}(alg.n_particles, alg.n_iters, alg.resampler, new_gid)

const CSMC = PG # type alias of PG as Conditional SMC

mutable struct PGInfo{Tidcs}
    logevidence::Vector{Float64}
    progress::ProgressMeter.Progress
    cache_updated::UInt8
    idcs::Tidcs
end

function Sampler(alg::PG, vi::AbstractVarInfo)
    idcs = VarReplay._getidcs(vi, Sampler(alg, nothing))
    info = PGInfo(Float64[], ProgressMeter.Progress(0, 1, "[PG] Sampling...", 0), 
                    CACHERESET, idcs)
    return Sampler(alg, info)
end
function init_spl(model, alg::PG; kwargs...)
    vi = VarInfo(model)
    spl = Sampler(alg, vi)
    return spl, vi
end


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
    push!(spl.info.logevidence, particles.logE)
    return particles[indx].vi, true
end

function init_samples(alg::PG, sample::Tsample, kwargs...) where {Tsample <: Sample}
    return Vector{Tsample}()
end
function init_varinfo(model, spl::Sampler{<:PG}; resume_from = nothing, kwargs...)
    if resume_from == nothing
        return VarInfo(model)
    else
        return resume_from.info.vi
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
        spl.info.progress = ProgressMeter.Progress(n, 1, "[PG] Sampling...", 0)
    end

    for i = 1:n
        time_elapsed = @elapsed vi, _ = step(model, spl, vi)
        push!(samples, Sample(vi))
        samples[i].info.elapsed = time_elapsed

        time_total += time_elapsed

        if PROGRESS[]  && spl.alg.gid == 0
            ProgressMeter.next!(spl.info.progress)
        end
    end

    @info("[PG] Finished with")
    @info("  Running time    = $time_total;")

    loge = exp.(mean(spl.info.logevidence))
    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = exp.(resume_from.logevidence)
        # Calculate new log-evidence
        pre_n = length(resume_from.info[:samples])
        loge = (log(pre_loge) * pre_n + log(loge) * n) / (pre_n + n)
    end
    c = Chain(loge, samples)       # wrap the result by Chain

    if save_state               # save state
        c = save(c, spl, model, vi, samples)
    end

    c
end

function assume(spl::Sampler{T}, dist::Distribution, vn::VarName, _::AbstractVarInfo) where T<:Union{PG,SMC}
    vi = current_trace().vi
    if isempty(getspace(spl)) || vn.sym in getspace(spl)
        if ~haskey(vi, vn)
            r = rand(dist)
            push!(vi, vn, r, dist, spl.alg.gid)
            spl.info.cache_updated = CACHERESET # sanity flag mask for getidcs and getranges
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
    r, 0.0
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
    0.0
end

function observe(spl::Sampler{A}, ds::Vector{D}, value::Any, vi::AbstractVarInfo) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing observe statement")
end
