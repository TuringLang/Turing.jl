function _sample(vi, samples, spl, model, alg::DynamicNUTS, chunk_size=CHUNKSIZE[]) where T <: Hamiltonian
    if ADBACKEND[] == :forward_diff
        default_chunk_size = CHUNKSIZE[]  # record global chunk size
        setchunksize(chunk_size)        # set temp chunk size
    end
    
    if spl.alg.gid == 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    function _lp(x)
        value, deriv = gradient(x, vi, model, spl)
        return ValueGradient(-value, -deriv)
    end

    chn_dynamic, _ = NUTS_init_tune_mcmc(FunctionLogDensity(length(vi[spl]), _lp), alg.n_iters)

    for i = 1:alg.n_iters
        vi[spl] = chn_dynamic[i].q
        samples[i].value = Sample(vi, spl).value
    end

    if ADBACKEND[] == :forward_diff
        setchunksize(default_chunk_size)      # revert global chunk size
    end

    return Chain(0, samples)
end
