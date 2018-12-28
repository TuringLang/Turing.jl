@inline function buildspace(subalgs, space=())
    length(subalgs) == 0 && return ()
    new_space = _filterspace(getspace(subalgs[1]), space)
    space = (space..., new_space...)
    return (new_space..., buildspace(Base.tail(subalgs), space)...)
end
@inline function _filterspace(new_space, space)
    length(new_space) == 0 && return ()
    if new_space[1] ∈ space
        return _filterspace(Base.tail(new_space), space)
    else
        return (new_space[1], _filterspace(Base.tail(new_space), space)...)
    end
end
_setdiff(superspace, subspace) = _filterspace(superspace, subspace)
function verifyspace(subalgs::Tuple{Vararg{InferenceAlgorithm}}, pvars, alg_str)
    space = buildspace(subalgs)
    verifyspace(space, pvars, alg_str)
end
function verifyspace(space::Tuple{Vararg{Symbol}}, pvars, alg_str)
    if length(space) > 0
        @assert issubset(pvars, space) "[$alg_str] symbols specified to samplers ($space) doesn't cover the model parameters $(Set(pvars))"

        if !(issubset(pvars, space) && issubset(space, pvars))
            @warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(_setdiff(space, pvars))")
        end
    end
    return 
end
