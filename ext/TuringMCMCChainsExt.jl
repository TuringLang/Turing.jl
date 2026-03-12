module TuringMCMCChainsExt

using Turing
using MCMCChains: MCMCChains

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
            this_walker_samples, model, spl, state, $Tchain; kwargs...
        )
    end
    return AbstractMCMC.chainscat(chains...)
end

end
