####
#### Stan interface
####

# NOTE:
#   Type fields
#     fieldnames(CmdStan.Sample)
#       num_samples, num_warmup, save_warmup, thin, adapt, algorithm

#     fieldnames(CmdStan.Hmc)
#       engine, metric, stepsize, stepsize_jitter

#     fieldnames(CmdStan.Adapt)
#       engaged, gamma, delta, kappa, t0, init_buffer, term_buffer, window

#   Ref
#     http://goedman.github.io/Stan.jl/latest/index.html#Types-1

function sample(mf, ss::CmdStan.Sample)
    return sample(mf, ss.num_samples, ss.num_warmup,
                    ss.save_warmup, ss.thin, ss.adapt, ss.alg)
end

function sample(
    mf,
    num_samples::Int,
    num_warmup::Int,
    save_warmup::Bool,
    thin::Int,
    ss::CmdStan.Sample
)
    return sample(mf, num_samples, num_warmup, save_warmup, thin, ss.adapt, ss.alg)
end

function sample(
    mf,
    num_samples::Int,
    num_warmup::Int,
    save_warmup::Bool,
    thin::Int,
    adapt::CmdStan.Adapt,
    alg::CmdStan.Hmc
)
    if alg.stepsize_jitter != 0.0
        @warn("[Turing.sample] Turing does not support adding noise to stepsize yet.")
    end
    if !adapt.engaged
        if isa(alg.engine, CmdStan.Static)   # hmc
            stepnum = alg.engine.int_time ÷ alg.stepsize
            sample(mf, HMC(alg.stepsize, stepnum), num_samples; adaptor=NUTSAdaptor(adapt))
        elseif isa(alg.engine, CmdStan.Nuts) # error
            error("[Turing.sample] CmdStan.Nuts cannot be used with adapt.engaged set as false")
        end
    else
        if isa(alg.engine, CmdStan.Static)   # hmcda
            sample(mf, HMCDA(num_warmup, adapt.delta, alg.engine.int_time), num_samples;
                    adaptor=NUTSAdaptor(adapt))
        elseif isa(alg.engine, CmdStan.Nuts) # nuts
            if isa(alg.metric, CmdStan.diag_e)
                sample(mf, NUTS(num_warmup, adapt.delta), num_samples;
                        adaptor=NUTSAdaptor(adapt))
            else # TODO: reove the following since Turing support this feature now.
                @warn("[Turing.sample] Turing does not support full covariance matrix for pre-conditioning yet.")
            end
        end
    end
end

# FIXME: where is `spl` in the code below?
function AHMCAdaptor(adaptor)
    if :engaged in fieldnames(typeof(adaptor)) # CmdStan.Adapt
        adaptor.engaged ? spl.alg.n_adapts : 0,
        AHMC.MassMatrixAdaptor(metric),
        AHMC.StepSizeAdaptor(adaptor.gamma,
            adaptor.t0, adaptor.kappa, adaptor.δ, init_ϵ),
            adaptor.init_buffer,
            adaptor.term_buffer,
            adaptor.window
    else # default adaptor
        @warn "Invalid adaptor type: $(typeof(adaptor)). Default adaptor is used instead."
        adaptor = AHMC.StanHMCAdaptor(
            AHMC.MassMatrixAdaptor(:DiagEuclideanMetric),
            AHMC.StepSizeAdaptor(spl.alg.δ, init_ϵ)
        )
        AHMC.initialize!(adaptor, spl.alg.n_adapts)
        adaptor
    end
end
