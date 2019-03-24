"""
    IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

- `n_particles::Int` : Number of particles to use.
- `n_iters::Int` : Number of iterations to employ.
- `n_nodes::Int` : The number of nodes running SMC and CSMC.
- `n_csmc_nodes::Int` : The number of CSMC nodes.

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    x[1] ~ Normal(m, sqrt(s))
    x[2] ~ Normal(m, sqrt(s))
    return s, m
end

sample(gdemo([1.5, 2]), IPMCMC(100, 100, 4, 2))
```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).
"""
mutable struct IPMCMC{space, F} <: InferenceAlgorithm
    n_particles           ::    Int         # number of particles used
    n_iters               ::    Int         # number of iterations
    n_nodes               ::    Int         # number of nodes running SMC and CSMC
    n_csmc_nodes          ::    Int         # number of nodes CSMC
    resampler             ::    F           # function to resample
    gid                   ::    Int         # group ID
end
function IPMCMC(n1::Int, n2::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, 32, 16, resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, n3, Int(ceil(n3/2)), resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int)
    F = typeof(resample_systematic)
    return IPMCMC{(), F}(n1, n2, n3, n4, resample_systematic, 0)
end
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int, space...)
    F = typeof(resample_systematic)
    IPMCMC{space, F}(n1, n2, n3, n4, resample_systematic, 0)
end
function IPMCMC(alg::IPMCMC, new_gid::Int)
    F = typeof(alg.resampler)
    @unpack n_particles, n_iters, n_nodes, n_csmc_nodes, resampler = alg
    S = getspace(alg)
    return IPMCMC{S, F}(n_particles, n_iters, n_nodes, n_csmc_nodes, resampler, new_gid)
end

mutable struct IPMCMCInfo{Tsamplers}
    samplers::Tsamplers
    progress::ProgressMeter.Progress
end
function IPMCMCInfo(samplers, alg::IPMCMC)
    progress = ProgressMeter.Progress(alg.n_iters, 1, "[IPMCMC] Sampling...", 0)
    return IPMCMCInfo(samplers, progress)
end

function Sampler(alg::IPMCMC, vi::Vector{<:AbstractVarInfo})
    # Create SMC and CSMC nodes
    # Use resampler_threshold=1.0 for SMC since adaptive resampling is invalid in this setting
    default_CSMC = CSMC(alg.n_particles, 1, alg.resampler, getspace(alg), 0)
    default_SMC = SMC(alg.n_particles, alg.resampler, 1.0, false, getspace(alg), 0)

    samplers1 = Tuple([Sampler(CSMC(default_CSMC, i), vi[i]) for i in 1:alg.n_csmc_nodes])
    samplers2 = Tuple([Sampler(SMC(default_SMC, i), vi[i]) for i in (alg.n_csmc_nodes+1):alg.n_nodes])
    info = IPMCMCInfo((samplers1..., samplers2...), alg)

    return Sampler(alg, info)
end

function init_spl(model::Model, alg::IPMCMC; kwargs...)
    vi = [VarInfo(model) for i in 1:alg.n_nodes]
    spl = Sampler(alg, vi)
    empty!.(vi)
    return spl, vi
end

function step(model, spl::Sampler{<:IPMCMC}, VarInfos::Array{<:VarInfo})
    # Initialise array for marginal likelihood estimators
    log_zs = zeros(spl.alg.n_nodes)

    # Run SMC & CSMC nodes
    for j in 1:spl.alg.n_nodes
        VarInfos[j].num_produce = 0
        VarInfos[j] = step(model, spl.info.samplers[j], VarInfos[j])[1]
        log_zs[j] = spl.info.samplers[j].info.logevidence[end]
    end

    # Resampling of CSMC nodes indices
    conditonal_nodes_indices = collect(1:spl.alg.n_csmc_nodes)
    unconditonal_nodes_indices = collect(spl.alg.n_csmc_nodes+1:spl.alg.n_nodes)
    for j in 1:spl.alg.n_csmc_nodes
        # Select a new conditional node by simulating cj
        log_ksi = vcat(log_zs[unconditonal_nodes_indices], log_zs[j])
        ksi = exp.(log_ksi .- maximum(log_ksi))
        c_j = wsample(ksi) # sample from Categorical with unormalized weights

        if c_j < length(log_ksi) # if CSMC node selects another index than itself
          conditonal_nodes_indices[j] = unconditonal_nodes_indices[c_j]
          unconditonal_nodes_indices[c_j] = j
        end
    end
    nodes_permutation = vcat(conditonal_nodes_indices, unconditonal_nodes_indices)

    return VarInfos[nodes_permutation]
end

get_sample_n(alg::IPMCMC; kwargs...) = alg.n_iters * alg.n_csmc_nodes

function _sample(VarInfos, samples, spl, model, alg::IPMCMC)
    # Init samples
    time_total = zero(Float64)
    # Init parameters
    n = spl.alg.n_iters

    # IPMCMC steps
    if PROGRESS[] 
        spl.info.progress = ProgressMeter.Progress(n, 1, "[IPMCMC] Sampling...", 0)
    end
    for i = 1:n
        Turing.DEBUG && @debug "IPMCMC stepping..."
        time_elapsed = @elapsed VarInfos = step(model, spl, VarInfos)

        # Save each CSMS retained path as a sample
        for j in 1:spl.alg.n_csmc_nodes
            samples[(i-1)*alg.n_csmc_nodes+j].value = Sample(VarInfos[j], spl).value
        end

        time_total += time_elapsed
        if PROGRESS[]
            isdefined(spl.info, :progress) && ProgressMeter.update!(spl.info.progress, spl.info.progress.counter + 1)
        end
    end

    println("[IPMCMC] Finished with")
    println("  Running time    = $time_total;")

    return Chain(0.0, samples) # wrap the result by Chain
end
