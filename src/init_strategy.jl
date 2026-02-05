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
    errmsg = "`initial_params` must be a `NamedTuple`, an `AbstractDict{<:VarName}`, or ideally a `DynamicPPL.AbstractInitStrategy`. Using a vector of parameters for `initial_params` is no longer supported. Please see https://turinglang.org/docs/usage/sampling-options/#specifying-initial-parameters for details on how to update your code."
    throw(ArgumentError(errmsg))
end
function _convert_initial_params(@nospecialize(_::Any))
    errmsg = "`initial_params` must be a `NamedTuple`, an `AbstractDict{<:VarName}`, or a `DynamicPPL.AbstractInitStrategy`."
    throw(ArgumentError(errmsg))
end

# TODO: Implement additional checks for certain samplers, e.g.
# HMC not supporting discrete parameters.
function _check_model(model::DynamicPPL.Model)
    new_model = DynamicPPL.setleafcontext(
        model, DynamicPPL.InitContext(DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll())
    )
    return DynamicPPL.check_model(
        new_model, DynamicPPL.OnlyAccsVarInfo(); error_on_failure=true
    )
end
function _check_model(model::DynamicPPL.Model, ::AbstractMCMC.AbstractSampler)
    return _check_model(model)
end
