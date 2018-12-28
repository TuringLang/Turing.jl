function step(model, spl::Sampler{<:IPMCMC}, VarInfos::Array{VarInfo}, is_first::Bool)
    # Initialise array for marginal likelihood estimators
    log_zs = zeros(spl.alg.n_nodes)

    # Run SMC & CSMC nodes
    for j in 1:spl.alg.n_nodes
        VarInfos[j].num_produce = 0
        VarInfos[j] = step(model, spl.info[:samplers][j], VarInfos[j])
        log_zs[j] = spl.info[:samplers][j].info[:logevidence][end]
    end

    # Resampling of CSMC nodes indices
    conditonal_nodes_indices = collect(1:spl.alg.n_csmc_nodes)
    unconditonal_nodes_indices = collect(spl.alg.n_csmc_nodes+1:spl.alg.n_nodes)
    for j in 1:spl.alg.n_csmc_nodes
        # Select a new conditional node by simulating cj
        log_ksi = vcat(log_zs[unconditonal_nodes_indices], log_zs[j])
        ksi = exp.(log_ksi-maximum(log_ksi))
        c_j = wsample(ksi) # sample from Categorical with unormalized weights

        if c_j < length(log_ksi) # if CSMC node selects another index than itself
          conditonal_nodes_indices[j] = unconditonal_nodes_indices[c_j]
          unconditonal_nodes_indices[c_j] = j
        end
    end
    nodes_permutation = vcat(conditonal_nodes_indices, unconditonal_nodes_indices)

    VarInfos[nodes_permutation]
end

get_sample_n(alg::IPMCMC; kwargs...) = alg.n_iters * alg.n_csmc_nodes

function init_varinfo(model, spl::Sampler{<:IPMCMC}; kwargs...)
    VarInfos = Array{VarInfo}(undef, spl.alg.n_nodes)
    for j in 1:spl.alg.n_nodes
        VarInfos[j] = VarInfo()
    end
    return VarInfos
end

function _sample(VarInfos, samples, spl, model, alg::IPMCMC)
    # Init samples
    time_total = zero(Float64)
    # Init parameters
    n = spl.alg.n_iters

    # IPMCMC steps
    if PROGRESS[] 
        spl.info[:progress] = ProgressMeter.Progress(n, 1, "[IPMCMC] Sampling...", 0)
    end
    for i = 1:n
        @debug "IPMCMC stepping..."
        time_elapsed = @elapsed VarInfos = step(model, spl, VarInfos, i==1)

        # Save each CSMS retained path as a sample
        for j in 1:spl.alg.n_csmc_nodes
            samples[(i-1)*alg.n_csmc_nodes+j].value = Sample(VarInfos[j], spl).value
        end

        time_total += time_elapsed
        if PROGRESS[]
            haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
        end
    end

    println("[IPMCMC] Finished with")
    println("  Running time    = $time_total;")

    Chain(0, samples) # wrap the result by Chain
end
