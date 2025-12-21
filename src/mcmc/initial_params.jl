"""
    find_initial_params(rng, model, varinfo, init_strategy, validator; max_attempts=10)

Attempt to find valid initial parameters for MCMC sampling.

This function tries to generate initial parameters that pass the provided validation function.
If the initial parameters are invalid, it will retry up to `max_attempts` times.

# Arguments
- `rng`: Random number generator
- `model`: The Turing model
- `varinfo`: Variable information structure
- `init_strategy`: Initialization strategy (passed to `DynamicPPL.init!!`)
- `validator`: Function that takes `varinfo` and returns `(is_valid::Bool, diagnostics::String)`

# Keyword Arguments
- `max_attempts::Int=1000`: Maximum number of attempts to find valid initial parameters

# Returns
- Valid `varinfo`

# Throws
- `ErrorException` if valid parameters cannot be found after `max_attempts` attempts

# Examples
```julia
# For external samplers with gradient checking
validator = vi -> begin
    θ = vi[:]
    try
        logp, grad = LogDensityProblems.logdensity_and_gradient(logdensity_f, θ)
        is_valid = isfinite(logp) && all(isfinite, grad)
        diagnostics = "logp=\$logp, grad_finite=\$(all(isfinite, grad))"
        return (is_valid, diagnostics)
    catch e
        return (false, "evaluation failed: \$e")
    end
end
varinfo = find_initial_params(rng, model, varinfo, init_strategy, validator)

# For HMC with hamiltonian
validator = vi -> begin
    θ = vi[:]
    z = AHMC.phasepoint(rng, θ, hamiltonian)
    is_valid = isfinite(z)
    diagnostics = "phasepoint finite: \$(isfinite(z))"
    return (is_valid, diagnostics)
end
varinfo = find_initial_params(rng, model, varinfo, init_strategy, validator)
```
"""
function find_initial_params(
    rng::AbstractRNG,
    model::Model,
    varinfo::AbstractVarInfo,
    init_strategy::DynamicPPL.AbstractInitStrategy,
    validator::Function;
    max_attempts::Int=1000,
)
    varinfo = deepcopy(varinfo)  # Don't mutate the input
    
    last_diagnostics = ""
    for attempt in 1:max_attempts
        # Validate current parameters
        is_valid, diagnostics = validator(varinfo)
        last_diagnostics = diagnostics
        
        if is_valid
            return varinfo  # Success!
        end
        
        # Warn at attempt 10
        if attempt == 10
            @warn "failed to find valid initial parameters in $(attempt) tries; consider providing a different initialisation strategy with the `initial_params` keyword"
        end
        
        # If this is the last attempt, throw informative error
        if attempt == max_attempts
            error(
                "Failed to find valid initial parameters after $max_attempts attempts. " *
                "Last attempt diagnostics: $last_diagnostics. " *
                "See https://turinglang.org/docs/uri/initial-parameters for common causes and solutions. " *
                "If the issue persists, please open an issue at https://github.com/TuringLang/Turing.jl/issues"
            )
        end
        
        # Regenerate parameters for next attempt
        _, varinfo = DynamicPPL.init!!(rng, model, varinfo, init_strategy)
    end
end