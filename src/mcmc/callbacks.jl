using AbstractMCMC: AbstractMCMC

function _get_lp(vi::DynamicPPL.AbstractVarInfo)
    lp = DynamicPPL.getlogp(vi)
    if lp isa NamedTuple
        return sum(values(lp))
    end
    return lp
end

function _varinfo_params(vi::DynamicPPL.AbstractVarInfo)
    vns = keys(vi)
    return Iterators.flatmap(vns) do vn
        val = DynamicPPL.getindex_internal(vi, vn)
        if val isa AbstractArray
            [string(vn, "[", i, "]") => v for (i, v) in enumerate(val)]
        else
            [string(vn) => val]
        end
    end
end

###
### getparams - Extract named parameters from sampler states
###

# HMCState - used by HMC, HMCDA, NUTS (contains vi field)
function AbstractMCMC.getparams(state::HMCState)
    return collect(_varinfo_params(state.vi))
end

# MHState - contains varinfo field (different name!)
function AbstractMCMC.getparams(state::MHState)
    return collect(_varinfo_params(state.varinfo))
end

# PGState - contains vi field
function AbstractMCMC.getparams(state::PGState)
    return collect(_varinfo_params(state.vi))
end

# SMCState - particles contain VarInfo in their model
function AbstractMCMC.getparams(state::SMCState)
    particle = state.particles.vals[state.particleindex]
    vi = particle.model.f.varinfo
    return collect(_varinfo_params(vi))
end

# GibbsState - contains global VarInfo
function AbstractMCMC.getparams(state::GibbsState)
    return collect(_varinfo_params(state.vi))
end

# ESS uses VarInfo directly as state
function AbstractMCMC.getparams(state::DynamicPPL.AbstractVarInfo)
    return collect(_varinfo_params(state))
end

# SGHMCState - contains params vector directly (no VarInfo)
function AbstractMCMC.getparams(state::SGHMCState)
    return ["θ[$i]" => v for (i, v) in enumerate(state.params)]
end

# SGLDState - contains params vector directly (no VarInfo)
function AbstractMCMC.getparams(state::SGLDState)
    return ["θ[$i]" => v for (i, v) in enumerate(state.params)]
end

###
### getstats - Extract extra statistics from sampler states
###

# HMCState - rich stats available
function AbstractMCMC.getstats(state::HMCState)
    lp = _get_lp(state.vi)
    # Get step size from kernel
    ϵ = try
        state.kernel.τ.integrator.ϵ
    catch
        NaN
    end
    return (lp=lp, step_size=ϵ, iteration=state.i)
end

# MHState
function AbstractMCMC.getstats(state::MHState)
    return (lp=state.logjoint_internal,)
end

# PGState
function AbstractMCMC.getstats(state::PGState)
    lp = _get_lp(state.vi)
    return (lp=lp,)
end

# SMCState
function AbstractMCMC.getstats(state::SMCState)
    return (logevidence=state.average_logevidence, particle_index=state.particleindex)
end

# GibbsState
function AbstractMCMC.getstats(state::GibbsState)
    lp = _get_lp(state.vi)
    return (lp=lp,)
end

# ESS (VarInfo as state)
function AbstractMCMC.getstats(state::DynamicPPL.AbstractVarInfo)
    lp = _get_lp(state)
    return (lp=lp,)
end

# SGHMCState
function AbstractMCMC.getstats(state::SGHMCState)
    lp = try
        LogDensityProblems.logdensity(state.logdensity, state.params)
    catch
        NaN
    end
    return (lp=lp,)
end

# SGLDState
function AbstractMCMC.getstats(state::SGLDState)
    lp = try
        LogDensityProblems.logdensity(state.logdensity, state.params)
    catch
        NaN
    end
    return (lp=lp, step=state.step)
end

###
### getparams/getstats from transitions (ParamsWithStats)
###

function AbstractMCMC.getparams(transition::DynamicPPL.ParamsWithStats)
    # params is OrderedDict{VarName, Any}
    return [string(vn) => val for (vn, val) in transition.params]
end

function AbstractMCMC.getstats(transition::DynamicPPL.ParamsWithStats)
    return transition.stats
end

###
### hyperparam_metrics - Define TensorBoard hyperparam metrics
###

function AbstractMCMC.hyperparam_metrics(model::DynamicPPL.Model, sampler::NUTS)
    return [
        "extras/acceptance_rate/stat/Mean",
        "extras/max_hamiltonian_energy_error/stat/Mean",
        "extras/lp/stat/Mean",
        "extras/n_steps/stat/Mean",
        "extras/tree_depth/stat/Mean",
    ]
end

function AbstractMCMC.hyperparam_metrics(model::DynamicPPL.Model, sampler::Hamiltonian)
    return [
        "extras/acceptance_rate/stat/Mean",
        "extras/lp/stat/Mean",
        "extras/n_steps/stat/Mean",
    ]
end

function AbstractMCMC.hyperparam_metrics(model::DynamicPPL.Model, sampler::MH)
    return ["extras/lp/stat/Mean"]
end

function AbstractMCMC.hyperparam_metrics(model::DynamicPPL.Model, sampler::PG)
    return ["extras/lp/stat/Mean", "extras/logevidence/stat/Mean"]
end

###
### _hyperparams_impl - Extract sampler hyperparameters
###

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::HMC, state; kwargs...
)
    return ["epsilon" => sampler.ϵ, "n_leapfrog" => sampler.n_leapfrog]
end

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::HMCDA, state; kwargs...
)
    return [
        "n_adapts" => sampler.n_adapts,
        "delta" => sampler.δ,
        "lambda" => sampler.λ,
        "epsilon" => sampler.ϵ,
    ]
end

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::NUTS, state; kwargs...
)
    return [
        "n_adapts" => sampler.n_adapts,
        "delta" => sampler.δ,
        "max_depth" => sampler.max_depth,
        "Delta_max" => sampler.Δ_max,
        "epsilon" => sampler.ϵ,
    ]
end

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::PG, state; kwargs...
)
    return ["nparticles" => sampler.nparticles]
end

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::SGHMC, state; kwargs...
)
    return [
        "learning_rate" => sampler.learning_rate, "momentum_decay" => sampler.momentum_decay
    ]
end

function AbstractMCMC._hyperparams_impl(
    model::DynamicPPL.Model, sampler::SGLD, state; kwargs...
)
    return ["stepsize" => string(sampler.stepsize)]
end
