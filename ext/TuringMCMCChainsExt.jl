module TuringMCMCChainsExt

using Turing
using Turing: AbstractMCMC, DynamicPPL
using Turing.Inference: HMC, NUTS, HMCDA, Emcee, EmceeState, SMC, _get_n_walkers
using MCMCChains: MCMCChains

import Turing.Inference: bundle_smc_samples, post_sample_hook

"""
    loadstate(chain::MCMCChains.Chains)

Load the final state of the sampler from a `MCMCChains.Chains` object.

To save the final state of the sampler, you must use `sample(...; save_state=true)`. If this
argument was not used during sampling, calling `loadstate` will throw an error.
"""
function Turing.Inference.loadstate(chain::MCMCChains.Chains)
    if !haskey(chain.info, :samplerstate)
        throw(
            ArgumentError(
                "the chain object does not contain the final state of the sampler; to save the final state you must sample with `save_state=true`",
            ),
        )
    end
    return chain.info[:samplerstate]
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:Vector},
    model::DynamicPPL.Model,
    spl::Emcee,
    state::EmceeState,
    ::Type{MCMCChains.Chains},
    kwargs...,
)
    n_walkers = _get_n_walkers(spl)
    chains = map(1:n_walkers) do i
        this_walker_samples = [s[i] for s in samples]
        AbstractMCMC.bundle_samples(
            this_walker_samples, model, spl, state, MCMCChains.Chains; kwargs...
        )
    end
    return AbstractMCMC.chainscat(chains...)
end

function bundle_smc_samples(
    transitions::Vector{<:DynamicPPL.ParamsWithStats},
    model::DynamicPPL.Model,
    sampler::SMC,
    state,
    ::Type{MCMCChains.Chains};
    kwargs...,
)
    # MCMCChains drops non-scalar statistics, so give each filtering step its own internal.
    ess_per_step = first(transitions).stats.ess_per_step
    ess_names = ntuple(i -> Symbol("ess_per_step[$i]"), length(ess_per_step))
    ess_stats = NamedTuple{ess_names}(Tuple(ess_per_step))
    scalar_transitions = map(transitions) do transition
        names = filter(!=(:ess_per_step), keys(transition.stats))
        stats = merge(NamedTuple{names}(transition.stats), ess_stats)
        DynamicPPL.ParamsWithStats(transition.params, stats)
    end
    return AbstractMCMC.bundle_samples(
        scalar_transitions, model, sampler, state, MCMCChains.Chains; kwargs...
    )
end

"""
    post_sample_hook(chain::MCMCChains.Chains, sampler::Union{HMC,NUTS,HMCDA}; kwargs...)

Emit a warning message if there are divergent transitions in the chain.
"""
function post_sample_hook(
    chain::MCMCChains.Chains, ::Union{HMC,NUTS,HMCDA}; verbose::Bool=true, kwargs...
)
    n_divergent = round(Int, sum(skipmissing(vec(chain[:numerical_error]))))
    if verbose && n_divergent > 0
        @warn "There were $n_divergent divergent transitions. Consider reparameterising your model or using a smaller step size. For adaptive samplers such as NUTS and HMCDA, consider increasing `target_accept`."
    end
    return nothing
end

end
