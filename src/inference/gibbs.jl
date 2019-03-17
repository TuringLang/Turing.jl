"""
    Gibbs(n_iters, algs...)

Compositional MCMC interface. Gibbs sampling combines one or more
sampling algorithms, each of which samples from a different set of
variables in a model.

Example:
```julia
@model gibbs_example(x) = begin
    v1 ~ Normal(0,1)
    v2 ~ Categorical(5)
        ...
end

# Use PG for a 'v2' variable, and use HMC for the 'v1' variable.
# Note that v2 is discrete, so the PG sampler is more appropriate
# than is HMC.
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```

Tips:
- `HMC` and `NUTS` are fast samplers, and can throw off particle-based
methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including
more particles in the particle sampler.
"""
mutable struct Gibbs{space, A} <: AbstractGibbs{space}
    n_iters   ::  Int     # number of Gibbs iterations
    algs      ::  A   # component sampling algorithms
    thin      ::  Bool    # if thinning to output only after a whole Gibbs sweep
    gid       ::  Int
end
function Gibbs(n_iters::Int, algs...; thin=true)
    Gibbs{buildspace(algs), typeof(algs)}(n_iters, algs, thin, 0)
end
Gibbs(alg::Gibbs, new_gid) = Gibbs(alg.n_iters, alg.algs, alg.thin, new_gid)

const GibbsComponent = Union{Hamiltonian,MH,PG}

@inline function get_gibbs_samplers(subalgs, model, n, alg, alg_str)
    if length(subalgs) == 0
        return (), ()
    else
        subalg = subalgs[1]
        if isa(subalg, GibbsComponent)
            spl, vi = init_spl(model, typeof(subalg)(subalg, n + 1 - length(subalgs)))
            spls_vis = get_gibbs_samplers(Base.tail(subalgs), model, n, alg, alg_str)
            spls = (spl, spls_vis[1]...)
            vis = (vi, spls_vis[2]...)
            return spls, vis
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
    end
end  

mutable struct GibbsInfo{Tsamplers}
    samplers::Tsamplers
    progress::ProgressMeter.Progress
end
function GibbsInfo(alg::InferenceAlgorithm, samplers)
    progress = ProgressMeter.Progress(alg.n_iters, 1, "[Gibbs] Sampling...", 0)
    return GibbsInfo(samplers, progress)
end

function Sampler(alg::Gibbs, model::Model)
    return init_spl(model, alg)[1]
end

function init_spl(model::Model, alg::Gibbs; kwargs...)
    n_samplers = length(alg.algs)
    alg_str = "Gibbs"
    samplers, vis = get_gibbs_samplers(alg.algs, model, n_samplers, alg, alg_str)
    vi = VarInfo(model)
    space = buildspace(alg.algs)
    verifyspace(space, model.pvars, alg_str)
    info = GibbsInfo(alg, samplers)
    spl = Sampler(alg, info)
    return spl, vi
end

function get_sample_n(alg::Gibbs; reuse_spl_n = 0, kwargs...)
    sub_sample_n = []
    for sub_alg in alg.algs
        if isa(sub_alg, GibbsComponent)
            push!(sub_sample_n, sub_alg.n_iters)
        else
            @error("[Gibbs] unsupport base sampling algorithm $alg")
        end
    end

    # Compute the number of samples to store
    n = reuse_spl_n > 0 ? reuse_spl_n : alg.n_iters
    sample_n = n * (alg.thin ? 1 : sum(sub_sample_n))

    return sample_n
end

macro gibbs_iteration(i)
    return esc(gibbs_iteration_expr(i))
end
function gibbs_iteration_expr(i)
    return quote
        Turing.DEBUG && @debug "Gibbs stepping..."

        time_elapsed = zero(Float64)
        lp = nothing; epsilon = nothing; lf_num = nothing; eval_num = nothing

        for local_spl in spl.info.samplers
            last_spl = local_spl

            Turing.DEBUG && @debug "$(typeof(local_spl)) stepping..."

            if isa(local_spl.alg, GibbsComponent)
                for _ = 1:local_spl.alg.n_iters
                    Turing.DEBUG && @debug "recording old θ..."
                    time_elapsed_thin = @elapsed varInfo, is_accept = step(model, local_spl, varInfo, Val($(i==1)))

                    if ~spl.alg.thin
                        samples[i_thin].value = Sample(varInfo).value
                        samples[i_thin].info.elapsed = time_elapsed_thin
                        if ~isa(local_spl.alg, Hamiltonian)
                            # If statement below is true if there is a HMC component which provides lp and epsilon
                            if lp != nothing
                                value = samples[i_thin].value
                                value = @set value.lp = lp
                                sample = samples[i_thin]
                                samples[i_thin] = @set sample.value = value
                            end
                            epsilon != nothing && (samples[i_thin].info.epsilon = epsilon)
                            lf_num != nothing && (samples[i_thin].info.lf_num = lf_num)
                            eval_num != nothing && (samples[i_thin].info.eval_num = eval_num)
                        end
                        i_thin += 1
                    end
                    time_elapsed += time_elapsed_thin
                end

                if isa(local_spl.alg, Hamiltonian)
                    lp = getlogp(varInfo)
                    epsilon = getss(local_spl.info.wum)
                    lf_num = local_spl.info.lf_num
                    eval_num = local_spl.info.eval_num
                end
            else
                @error("[Gibbs] unsupport base sampler $local_spl")
            end
        end

        time_total += time_elapsed

        if spl.alg.thin
            samples[$i].value = Sample(varInfo).value
            samples[$i].info.elapsed = time_elapsed
            # If statement below is true if there is a HMC component which provides lp and epsilon
            if lp != nothing
                value = samples[$i].value
                value = @set value.lp = lp
                sample = samples[$i]
                samples[$i] = @set sample.value = value
            end
            epsilon != nothing && (samples[$i].info.epsilon = epsilon)
            lf_num != nothing && (samples[$i].info.lf_num = lf_num)
            eval_num != nothing && (samples[$i].info.eval_num = eval_num)
        end

        if PROGRESS[]
            if isdefined(spl.info, :progress)
                ProgressMeter.update!(spl.info.progress, spl.info.progress.counter + 1)
            end
        end
    end
end

function _sample(varInfo,
                samples,
                spl,
                model,
                alg::Gibbs;
                save_state=false,         # flag for state saving
                resume_from=nothing,      # chain to continue
                reuse_spl_n=0,             # flag for spl re-using
                )

    # Init samples
    time_total = zero(Float64)
    n = spl.alg.n_iters; i_thin = 1
    # Gibbs steps
    PROGRESS[] && (spl.info.progress = ProgressMeter.Progress(n, 1, "[Gibbs] Sampling...", 0))
    @gibbs_iteration(1)
    for i = 2:n
        @gibbs_iteration(i)
    end

    @info("[Gibbs] Finished with")
    @info("  Running time    = $time_total;")

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
    end
    c = Chain(0.0, samples)       # wrap the result by Chain

    if save_state               # save state
        c = save(c, spl, model, varInfo, samples)
    end

    return c
end
