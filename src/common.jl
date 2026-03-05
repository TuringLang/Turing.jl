using AbstractPPL: VarName
using DynamicPPL: DynamicPPL

# These functions are shared by both MCMC and optimisation, so has to exist outside of both.

"""
    _convert_initial_params(initial_params)

Convert `initial_params` to a `DynamicPPl.AbstractInitStrategy` if it is not already one, or
throw a useful error message.
"""
_convert_initial_params(initial_params::DynamicPPL.AbstractInitStrategy) = initial_params
function _convert_initial_params(nt::NamedTuple)
    @info "Using a NamedTuple for `initial_params` will be deprecated in a future release. Please use `InitFromParams(namedtuple)` instead."
    return DynamicPPL.InitFromParams(nt)
end
function _convert_initial_params(d::AbstractDict{<:VarName})
    @info "Using a Dict for `initial_params` will be deprecated in a future release. Please use `InitFromParams(dict)` instead."
    return DynamicPPL.InitFromParams(d)
end
function _convert_initial_params(::AbstractVector{<:Real})
    errmsg = "`initial_params` must be an `DynamicPPL.AbstractInitStrategy`. Using a vector of parameters for `initial_params` is no longer supported. Please see https://turinglang.org/docs/usage/sampling-options/#specifying-initial-parameters for details on how to update your code."
    throw(ArgumentError(errmsg))
end
function _convert_initial_params(@nospecialize(_::Any))
    errmsg = "`initial_params` must be a `DynamicPPL.AbstractInitStrategy`."
    throw(ArgumentError(errmsg))
end

allow_discrete_variables(sampler::AbstractMCMC.AbstractSampler) = true
function _check_model(model::DynamicPPL.Model, fail_if_discrete::Bool)
    result = DynamicPPL.check_model(
        model; error_on_failure=false, fail_if_discrete=fail_if_discrete
    )
    if !result
        throw(
            ArgumentError(
                "The model $(model.f) has one or more issues that may cause inference to fail. Please see the warnings above for details.\n\nIf you think that this is a false positive, you can disable this by passing the `check_model=false` keyword argument to `sample` or the mode estimation functions. Please also consider opening an issue.\n",
            ),
        )
    end
end
function _check_model(model::DynamicPPL.Model, sampler::AbstractMCMC.AbstractSampler)
    # This is hit by MCMC
    return _check_model(model, !allow_discrete_variables(sampler))
end
function _check_model(model::DynamicPPL.Model)
    # Optimisation hits this. TODO: We allow discrete variables now, but that does depend on
    # the optimisation algorithm, surely?
    return _check_model(model, false)
end

# Similar to InitFromParams, this is just for convenience
_to_varnamedtuple(nt::NamedTuple) = DynamicPPL.VarNamedTuple(nt)
_to_varnamedtuple(d::AbstractDict{<:VarName}) = DynamicPPL.VarNamedTuple(pairs(d))
_to_varnamedtuple(vnt::DynamicPPL.VarNamedTuple) = vnt
