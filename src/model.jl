"""
    struct Model{pvars, dvars, F, TD}
        f::F
        data::TD
    end
    
A `Model` struct with parameter variables `pvars`, data variables `dvars`, inner 
function `f` and `data::NamedTuple`.
"""
struct Model{pvars, dvars, F, TD}
    f::F
    data::TD
end
function Model{pvars, dvars}(f::F, data::TD) where {pvars, dvars, F, TD}
    return Model{pvars, dvars, F, TD}(f, data)
end
pvars(m::Model{params}) where {params} = Tuple(params.types)
dvars(m::Model{params, data}) where {params, data} = Tuple(data.types)
function Base.getproperty(m::Model, f::Symbol)
    f === :pvars && return pvars(m)
    return getfield(m, f)
end

@generated function inpvars(::Val{sym}, ::Model{params}) where {sym, params}
    return sym in params.types ? :(true) : :(false)
end
@generated function indvars(::Val{sym}, ::Model{params, data}) where {sym, params, data}
    return sym in data.types ? :(true) : :(false)
end

(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)

function runmodel!(model, vi, spl)
    setlogp!(vi, zero(Real))
    if spl != nothing && :eval_num ∈ keys(spl.info)
        spl.info[:eval_num] += 1
    end
    model(vi, spl)
    return vi
end
