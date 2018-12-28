using Base.Meta: parse

#################
# Overload of ~ #
#################

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
@generated function inpvars(::Val{sym}, ::Model{params}) where {sym, params}
    return sym in params.types ? :(true) : :(false)
end
@generated function indvars(::Val{sym}, ::Model{params, data}) where {sym, params, data}
    return sym in data.types ? :(true) : :(false)
end

(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)

# TODO: Replace this macro, see issue #514
"""
Usage: @VarName x[1,2][1+5][45][3]
  return: (:x,[1,2],6,45,3)
"""
macro VarName(expr::Union{Expr, Symbol})
    ex = deepcopy(expr)
    isa(ex, Symbol) && return var_tuple(ex)
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    inds = :(())
    while ex.head == :ref
        if length(ex.args) >= 2
            pushfirst!(inds.args, Expr(:vect, ex.args[2:end]...))
            end
        ex = ex.args[1]
        isa(ex, Symbol) && return var_tuple(ex, inds)
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end
function var_tuple(sym::Symbol, inds::Expr=:(()))
    return esc(:($(QuoteNode(sym)), $inds, $(QuoteNode(gensym()))))
end


function wrong_dist_errormsg(l)
    return "Right-hand side of a ~ must be subtype of Distribution or a vector of" * 
        "Distributions on line $(l)."
end

"""
    generate_observe(observation, dist, model_info)

Generate an observe expression for observation `observation` drawn from 
a distribution or a vector of distributions (`dist`).
"""
function generate_observe(observation, dist, model_info)
    main_body_names = model_info[:main_body_names]
    vi = main_body_names[:vi]
    sampler = main_body_names[:sampler]
    return quote
        isdist = if isa($dist, AbstractVector)
            # Check if the right-hand side is a vector of distributions.
            all(d -> isa(d, Distribution), $dist)
        else
            # Check if the right-hand side is a distribution.
            isa($dist, Distribution)
        end
        @assert isdist @error($(wrong_dist_errormsg(@__LINE__)))
        $vi.logp += Turing.observe($sampler, $dist, $observation, $vi)
    end
end

"""
    generate_assume(var, dist, model_info)

Generate an assume expression for parameters `var` drawn from 
a distribution or a vector of distributions (`dist`).
"""
function generate_assume(var::Union{Symbol, Expr}, dist, model_info)
    main_body_names = model_info[:main_body_names]
    vi = main_body_names[:vi]
    sampler = main_body_names[:sampler]
    
    varname = gensym(:varname)
    sym, idcs, csym = gensym(:sym), gensym(:idcs), gensym(:csym)
    csym_str, indexing, syms = gensym(:csym_str), gensym(:indexing), gensym(:syms)
    
    if var isa Symbol
        varname_expr = quote
            $sym, $idcs, $csym = @VarName $var
            $csym = Symbol($(model_info[:name]), $csym)
            $syms = Symbol[$csym, $(QuoteNode(var))]
            $varname = Turing.VarName($vi, $syms, "")
        end
    else
        varname_expr = quote
            $sym, $idcs, $csym = @VarName $var
            $csym_str = string($(model_info[:name]))*string($csym)
            $indexing = mapfoldl(string, *, $idcs, init = "")
            $varname = Turing.VarName($vi, Symbol($csym_str), $sym, $indexing)
        end
    end
    
    lp = gensym(:lp)
    return quote
        $varname_expr
        isdist = if isa($dist, AbstractVector)
            # Check if the right-hand side is a vector of distributions.
            all(d -> isa(d, Distribution), $dist)
        else
            # Check if the right-hand side is a distribution.
            isa($dist, Distribution)
        end
        @assert isdist @error($(wrong_dist_errormsg(@__LINE__)))

        ($var, $lp) = if isa($dist, AbstractVector)
            Turing.assume($sampler, $dist, $varname, $var, $vi)
        else
            Turing.assume($sampler, $dist, $varname, $vi)
        end
        $vi.logp += $lp
    end
end

"""
    tilde(left, right, model_info)

The `tilde` function generates observation expression for data variables and assumption expressions for parameter variables, updating `model_info` in the process.
"""
function tilde(left, right, model_info)
    return generate_observe(left, right, model_info)
end
function tilde(left::Union{Symbol, Expr}, right, model_info)
    return _tilde(getvsym(left), left, right, model_info)
end

function _tilde(vsym, left, dist, model_info)
    main_body_names = model_info[:main_body_names]
    model_name = main_body_names[:model]

    if vsym in model_info[:arg_syms]
        if !(vsym in model_info[:tent_dvars_list])
            @debug " Observe - `$(vsym)` is an observation"
            push!(model_info[:tent_dvars_list], vsym)
        end

        return quote 
            if Turing.indvars($(Val(vsym)), $model_name)
                $(generate_observe(left, dist, model_info))
            else
                $(generate_assume(left, dist, model_info))
            end
        end
    else
        # Assume it is a parameter.
        if !(vsym in model_info[:tent_pvars_list])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @debug msg
            push!(model_info[:tent_pvars_list], vsym)
        end

        return generate_assume(left, dist, model_info)
    end
end

#################
# Main Compiler #
#################

"""
    @model(body)

Macro to specify a probabilistic model.

Example:

Model definition:

```julia
@model model_generator(x = default_x, y) = begin
    ...
end
```

Expanded model definition

```julia
# Allows passing arguments as kwargs
model_generator(; x = nothing, y = nothing)) = model_generator(x, y)
function model_generator(x = nothing, y = nothing)
    pvars, dvars = Turing.get_vars(Tuple{:x, :y}, (x = x, y = y))
    data = Turing.get_data(dvars, (x = x, y = y))
    
    inner_function(sampler::Turing.AnySampler, model) = inner_function(model)
    function inner_function(model)
        return inner_function(Turing.VarInfo(), Turing.SampleFromPrior(), model)
    end
    function inner_function(vi::Turing.VarInfo, model)
        return inner_function(vi, Turing.SampleFromPrior(), model)
    end
    # Define the main inner function
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AnySampler, model)
        local x
        if isdefined(model.data, :x)
            x = model.data.x
        else
            x = default_x
        end
        local y
        if isdefined(model.data, :y)
            y = model.data.y
        else
            y = nothing
        end

        vi.logp = zero(Real)
        ...
    end
    model = Turing.Model{pvars, dvars}(inner_function, data)
    return model
end
```

Generating a model: `model_generator(x_value)::Model`.
"""
macro model(input_expr)
    build_model_info(input_expr) |> translate_tilde! |> update_args! |> build_output
end

"""
    build_model_info(input_expr)

Builds the `model_info` dictionary from the model's expression.
"""
function build_model_info(input_expr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(input_expr)
    # Function body of the model is empty
    warn_empty(modeldef[:body])
    # Construct model_info dictionary
    
    arg_syms = [(arg isa Symbol) ? arg : arg.args[1] for arg in modeldef[:args]]
    model_info = Dict(
        :name => modeldef[:name],
        :input_expr => input_expr,
        :main_body => modeldef[:body],
        :arg_syms => arg_syms,
        :args => modeldef[:args],
        :kwargs => modeldef[:kwargs],
        :tent_dvars_list => Symbol[],
        :tent_pvars_list => Symbol[],
        :main_body_names => Dict(
            :vi => gensym(:vi), 
            :sampler => gensym(:sampler),
            :model => gensym(:model),
            :pvars => gensym(:pvars),
            :dvars => gensym(:dvars),
            :data => gensym(:data),
            :inner_function => gensym(:inner_function)
        )
    )

    return model_info
end

"""
    translate_tilde!(model_info)

Translates ~ expressions to observation or assumption expressions, updating `model_info`.
"""
function translate_tilde!(model_info)
    ex = model_info[:main_body]
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, model_info) : x, ex)
    model_info[:main_body] = ex
    return model_info
end

"""
    update_args!(model_info)

Extracts default argument values and replaces them with `nothing`.
"""
function update_args!(model_info)
    fargs = model_info[:args]
    fargs_default_values = Dict()
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            fargs_default_values[fargs[i]] = :nothing
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        elseif isa(fargs[i], Expr) && fargs[i].head == :kw
            fargs_default_values[fargs[i].args[1]] = fargs[i].args[2]
            fargs[i] = Expr(:kw, fargs[i].args[1], :nothing)
        else
            throw("Unsupported argument type $(fargs[i]).")
        end
    end
    model_info[:args] = fargs
    model_info[:arg_defaults] = fargs_default_values

    return model_info
end

"""
    build_output(model_info)

Builds the output expression.
"""
function build_output(model_info)
    # Construct user-facing function
    main_body_names = model_info[:main_body_names]
    vi_name = main_body_names[:vi]
    model_name = main_body_names[:model]
    sampler_name = main_body_names[:sampler]
    data_name = main_body_names[:data]
    pvars_name = main_body_names[:pvars]
    dvars_name = main_body_names[:dvars]
    inner_function_name = main_body_names[:inner_function]

    args = model_info[:args]
    arg_syms = model_info[:arg_syms]
    outer_function_name = model_info[:name]
    tent_pvars_list = model_info[:tent_pvars_list]
    tent_dvars_list = model_info[:tent_dvars_list]
    main_body = model_info[:main_body]
    arg_defaults = model_info[:arg_defaults]

    if length(tent_dvars_list) == 0
        tent_dvars_nt = :(NamedTuple())
    else
        tent_dvars_nt = :($([:($var = $var) for var in tent_dvars_list]...),)
    end

    #= Does the following for each of the tentative dvars
        local x
        if isdefined(model.data, :x)
            x = model.data.x
        else
            x = default_x
        end
    =#
    unwrap_data_expr = Expr(:block)
    for var in tent_dvars_list
        push!(unwrap_data_expr.args, quote
            local $var
            if isdefined($model_name.data, $(QuoteNode(var)))
                $var = $model_name.data.$var
            else
                $var = $(arg_defaults[var])
            end
        end)
    end

    return esc(quote
        # Allows passing arguments as kwargs
        $outer_function_name(;$(args...)) = $outer_function_name($(arg_syms...))
        # Outer function with `nothing` as default values
        function $outer_function_name($(args...))
            # Adds variables equal to `nothing` to pvars and the rest to dvars
            # `tent_pvars_list` is the tentative list of pvars
            # `tent_dvars_nt` is the tentative named tuple of dvars
            $pvars_name, $dvars_name = Turing.get_vars($(Tuple{tent_pvars_list...}), $(tent_dvars_nt))
            # Filter out the dvars equal to `nothing`
            $data_name = Turing.get_data($dvars_name, $tent_dvars_nt)
            
            # Define fallback inner functions
            function $inner_function_name($sampler_name::Turing.AnySampler, $model_name)
                return $inner_function_name($model_name)
            end
            function $inner_function_name($model_name)
                return $inner_function_name(Turing.VarInfo(), Turing.SampleFromPrior(), $model_name)
            end
            function $inner_function_name($vi_name::Turing.VarInfo, $model_name)
                return $inner_function_name($vi_name, Turing.SampleFromPrior(), $model_name)
            end

            # Define the main inner function
            function $inner_function_name(
                $vi_name::Turing.VarInfo, 
                $sampler_name::Turing.AnySampler, 
                $model_name
                )
                
                $unwrap_data_expr
                $vi_name.logp = zero(Real)
                $main_body
            end
            $model_name = Turing.Model{$pvars_name, $dvars_name}($inner_function_name, $data_name)
            return $model_name
        end
    end)
end

@generated function get_vars(tent_pvars::Type{Tpvars}, tent_dvars_nt::NamedTuple) where {Tpvars <: Tuple}
    tent_pvar_syms = [Tpvars.types...]
    tent_dvar_syms = [tent_dvars_nt.names...]
    dvar_types = [tent_dvars_nt.types...]
    append!(tent_pvar_syms, [tent_dvar_syms[i] for i in 1:length(tent_dvar_syms) if dvar_types[i] == Nothing])
    setdiff!(tent_dvar_syms, tent_pvar_syms)    
    pvars_tuple = Tuple{tent_pvar_syms...}
    dvars_tuple = Tuple{tent_dvar_syms...}

    return :($pvars_tuple, $dvars_tuple)
end

@generated function get_data(::Type{Tdvars}, nt) where Tdvars
    dvars = Tdvars.types
    args = []
    for var in dvars
        push!(args, :($var = nt.$var))
    end
    if length(args) == 0
        return :(NamedTuple())
    else
        return :($(args...),)
    end
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return 
end

####################
# Helper functions #
####################

getvsym(s::Symbol) = s
function getvsym(expr::Expr)
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    return getvsym(expr.args[1])
end
