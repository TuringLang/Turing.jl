function step(model, spl::Sampler{<:PMMH}, vi::AbstractVarInfo, is_first::Bool)
    violating_support = false
    proposal_ratio = 0.0
    new_prior_prob = 0.0
    new_likelihood_estimate = 0.0
    old_θ = copy(vi[spl])

    @debug "Propose new parameters from proposals..."
    for local_spl in spl.info[:samplers][1:end-1]
        @debug "$(typeof(local_spl)) proposing $(getspace(local_spl))..."
        propose(model, local_spl, vi)
        if local_spl.info[:violating_support] 
            violating_support = true
            break 
        end
        new_prior_prob += local_spl.info[:prior_prob]
        proposal_ratio += local_spl.info[:proposal_ratio]
    end

    if !violating_support # do not run SMC if going to refuse anyway
        @debug "Propose new state with SMC..."
        vi = step(model, spl.info[:samplers][end], vi)
        new_likelihood_estimate = spl.info[:samplers][end].info[:logevidence][end]

        @debug "computing accept rate α..."
        is_accept, logα = mh_accept(
            -(spl.info[:old_likelihood_estimate] + spl.info[:old_prior_prob]),
            -(new_likelihood_estimate + new_prior_prob),
            proposal_ratio,
        )
    end

    @debug "decide whether to accept..."
    if !violating_support && is_accept # accepted
        is_accept = true
        spl.info[:old_likelihood_estimate] = new_likelihood_estimate
        spl.info[:old_prior_prob] = new_prior_prob
    else                      # rejected
        is_accept = false
        vi[spl] = old_θ
    end

    return vi, is_accept
end

function _sample(vi, samples, spl, model, alg::PMMH;
                    save_state=false,         # flag for state saving
                    resume_from=nothing,      # chain to continue
                    reuse_spl_n=0             # flag for spl re-using
                )

    alg_str = "PMMH"
    # Init samples
    time_total = zero(Float64)
    # Init parameters
    n = spl.alg.n_iters

    # PMMH steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
        @debug "$alg_str stepping..."
        time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, i==1)

        if is_accept # accepted => store the new predcits
            samples[i].value = Sample(vi, spl).value
        else         # rejected => store the previous predcits
            samples[i] = samples[i - 1]
        end

        time_total += time_elapsed
        push!(accept_his, is_accept)
        if PROGRESS[]
            haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
        end
    end

    println("[$alg_str] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    if save_state               # save state
        save!(c, spl, model, vi)
    end

    c
end
