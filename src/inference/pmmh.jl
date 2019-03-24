"""
    PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})

Particle independant Metropolis–Hastings and
Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

- `n_iters::Int` : Number of iterations to run.
- `smc_alg:::SMC` : An [`SMC`](@ref) algorithm to use.
- `parameters_algs::Tuple{MH}` : An [`MH`](@ref) algorithm, which includes a
sample space specification.
"""
mutable struct PMMH{space, A<:Tuple} <: InferenceAlgorithm
    n_iters               ::    Int               # number of iterations
    algs                  ::    A                 # Proposals for state & parameters
    gid                   ::    Int               # group ID
end
PMMH{space}(n, algs, gid) where space = PMMH{space, typeof(algs)}(n, algs, gid)
function PMMH(n_iters::Int, smc_alg::SMC, parameter_algs...)
    algs = tuple(parameter_algs..., smc_alg)
    return PMMH{buildspace(algs)}(n_iters, algs, 0)
end
PMMH(alg::PMMH, new_gid) = PMMH{getspace(alg)}(alg.n_iters, alg.algs, new_gid)

function PIMH(n_iters::Int, smc_alg::SMC)
    algs = tuple(smc_alg)
    return PMMH{buildspace(algs)}(n_iters, algs, 0)
end

@inline function get_pmmh_samplers(subalgs, model, n, alg, alg_str)
    if length(subalgs) == 0
        return ()
    else
        subalg = subalgs[1]
        if typeof(subalg) == MH && subalg.n_iters != 1
            warn("[$alg_str] number of iterations greater than 1 is useless for MH since it is only used for its proposal")
        end
        if isa(subalg, Union{SMC, MH})
            spl, vi = init_spl(model, typeof(subalg)(subalg, n + 1 - length(subalgs)))
            _spls = get_pmmh_samplers(Base.tail(subalgs), model, n, alg, alg_str)
            spls = (spl, _spls...)
            return spls
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
    end
end  

mutable struct PMMHInfo{Tsamplers, Tidcs, Tranges}
    samplers::Tsamplers
    violating_support::Bool
    prior_prob::Float64
    proposal_ratio::Float64
    old_likelihood_estimate::Float64
    old_prior_prob::Float64
    progress::ProgressMeter.Progress
    cache_updated::UInt8
    idcs::Tidcs
    ranges::Tranges
end
function PMMHInfo(samplers, alg::PMMH, vi)
    spl = Sampler(alg, nothing)
    idcs = VarReplay._getidcs(vi, spl)
    ranges = VarReplay._getranges(vi, spl, idcs)
    n = alg.n_iters
    return PMMHInfo(samplers, false, 0.0, 0.0, -Inf, 0.0, ProgressMeter.Progress(n, 1, "[PMMH] Sampling...", 0), CACHERESET, idcs, ranges)
end

function init_spl(model, alg::PMMH; resume_from = nothing, kwargs...)
    alg_str = "PMMH"
    n_samplers = length(alg.algs)
    samplers = get_pmmh_samplers(alg.algs, model, n_samplers, alg, alg_str)
    verifyspace(alg.algs, model.pvars, alg_str)
    vi = if resume_from == nothing
        vi = empty!(VarInfo(model))
        model(vi, SampleFromUniform())
        vi
    else
        resume_from.info.vi
    end
    info = PMMHInfo(samplers, alg, vi)
    spl = Sampler(alg, info)
    return spl, vi
end

@inline function _step(samplers::Tuple, model, vi, violating_support, new_prior_prob, proposal_ratio)
    if length(samplers) == 1
        return violating_support, new_prior_prob, proposal_ratio
    end
    local_spl = samplers[1]
    propose(model, local_spl, vi)
    Turing.DEBUG && @debug "$(typeof(local_spl)) proposing $(getspace(local_spl))..."
    if local_spl.info.violating_support 
        violating_support = true
        return violating_support, new_prior_prob, proposal_ratio
    end
    new_prior_prob += local_spl.info.prior_prob
    proposal_ratio += local_spl.info.proposal_ratio

    return _step(Base.tail(samplers), model, vi, violating_support, new_prior_prob, proposal_ratio)
end

function step(model, spl::Sampler{<:PMMH}, vi::AbstractVarInfo)
    violating_support = false
    proposal_ratio = 0.0
    new_prior_prob = 0.0
    new_likelihood_estimate = 0.0
    old_θ = copy(vi[spl])

    Turing.DEBUG && @debug "Propose new parameters from proposals..."

    violating_support, new_prior_prob, proposal_ratio = 
        _step(spl.info.samplers, model, vi, violating_support, new_prior_prob, proposal_ratio)

    if !violating_support # do not run SMC if going to refuse anyway
        Turing.DEBUG && @debug "Propose new state with SMC..."
        vi, _ = step(model, spl.info.samplers[end], vi)
        new_likelihood_estimate = spl.info.samplers[end].info.logevidence[end]

        Turing.DEBUG && @debug "computing accept rate α..."
        is_accept, logα = mh_accept(
            -(spl.info.old_likelihood_estimate + spl.info.old_prior_prob),
            -(new_likelihood_estimate + new_prior_prob),
            proposal_ratio,
        )
    end

    Turing.DEBUG && @debug "decide whether to accept..."
    if !violating_support && is_accept # accepted
        is_accept = true
        spl.info.old_likelihood_estimate = new_likelihood_estimate
        spl.info.old_prior_prob = new_prior_prob
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
    PROGRESS[] && (spl.info.progress = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
        Turing.DEBUG && @debug "$alg_str stepping..."
        time_elapsed = @elapsed vi, is_accept = step(model, spl, vi)

        if is_accept # accepted => store the new predcits
            samples[i].value = Sample(vi, spl).value
        else         # rejected => store the previous predcits
            samples[i] = samples[i - 1]
        end

        time_total += time_elapsed
        push!(accept_his, is_accept)
        if PROGRESS[]
            isdefined(spl.info, :progress) && ProgressMeter.update!(spl.info.progress, spl.info.progress.counter + 1)
        end
    end

    println("[$alg_str] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
    end
    c = Chain(-Inf, samples)       # wrap the result by Chain

    if save_state               # save state
        save!(c, spl, model, vi, samples)
    end

    return c
end
