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
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[Gibbs] Sampling...", 0))
    for i = 1:n
        @debug "Gibbs stepping..."

        time_elapsed = zero(Float64)
        lp = nothing; epsilon = nothing; lf_num = nothing; eval_num = nothing

        for local_spl in spl.info[:samplers]
            last_spl = local_spl

            @debug "$(typeof(local_spl)) stepping..."

            if isa(local_spl.alg, GibbsComponent)
                for _ = 1:local_spl.alg.n_iters
                    @debug "recording old θ..."
                    time_elapsed_thin = @elapsed varInfo, is_accept = step(model, local_spl, varInfo, Val(i==1))

                    if ~spl.alg.thin
                        samples[i_thin].value = Sample(varInfo).value
                        samples[i_thin].value[:elapsed] = time_elapsed_thin
                        if ~isa(local_spl.alg, Hamiltonian)
                            # If statement below is true if there is a HMC component which provides lp and epsilon
                            if lp != nothing samples[i_thin].value[:lp] = lp end
                            if epsilon != nothing samples[i_thin].value[:epsilon] = epsilon end
                            if lf_num != nothing samples[i_thin].value[:lf_num] = lf_num end
                            if eval_num != nothing samples[i_thin].value[:eval_num] = eval_num end
                        end
                        i_thin += 1
                    end
                    time_elapsed += time_elapsed_thin
                end

                if isa(local_spl.alg, Hamiltonian)
                    lp = getlogp(varInfo)
                    epsilon = getss(local_spl.info[:wum])
                    lf_num = local_spl.info[:lf_num]
                    eval_num = local_spl.info[:eval_num]
                end
            else
                @error("[Gibbs] unsupport base sampler $local_spl")
            end
        end

        time_total += time_elapsed

        if spl.alg.thin
            samples[i].value = Sample(varInfo).value
            samples[i].value[:elapsed] = time_elapsed
            # If statement below is true if there is a HMC component which provides lp and epsilon
            if lp != nothing samples[i].value[:lp] = lp end
            if epsilon != nothing samples[i].value[:epsilon] = epsilon end
            if lf_num != nothing samples[i].value[:lf_num] = lf_num end
            if eval_num != nothing samples[i].value[:eval_num] = eval_num end
        end

        if PROGRESS[]
            if haskey(spl.info, :progress)
                ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
            end
        end
    end

    @info("[Gibbs] Finished with")
    @info("  Running time    = $time_total;")

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    if save_state               # save state
        save!(c, spl, model, varInfo)
    end

    return c
end
