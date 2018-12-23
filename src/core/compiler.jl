using Base.Meta: parse

#################
# Overload of ~ #
#################

struct Model{pvars, dvars, F, TD}
    f::F
    data::TD
end
function Model{pvars, dvars}(f::F, data::TD) where {pvars, dvars, F, TD}
    return Model{pvars, dvars, F, TD}(f, data)
end
pvars(m::Model{params}) where {params} = Tuple(params.types)
function dvars(m::Model{params, data}) where {params, data}
    return Tuple(data.types)
end
@generated function inpvars(::Val{sym}, ::Model{params}) where {sym, params}
    if sym in params.types
        return :(true)
    else
        return :(false)
    end
end
@generated function indvars(::Val{sym}, ::Model{params, data}) where {sym, params, data}
    if sym in data.types
        return :(true)
    else
        return :(false)
    end
end

(m::Model)(args...; kwargs...) = m.f(args..., m; kwargs...)

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


wrong_dist_errormsg(l) = "Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions on line $(l)."

"""
    generate_observe(observation, distribution, model_info)

Generate an observe expression for observation `observation` drawn from 
a distribution or a vector of distributions (`distribution`).
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
    generate_assume(variable, distribution, model_info)

Generate an assume expression for parameters `variable` drawn from 
a distribution or a vector of distributions (`distribution`).

"""
function generate_assume(var::Union{Symbol, Expr}, dist, model_info)
    main_body_names = model_info[:main_body_names]
    vi = main_body_names[:vi]
    sampler = main_body_names[:sampler]
    
    varname = gensym()
    sym = gensym(); idcs = gensym(); csym = gensym(); 
    csym_str = gensym(); indexing = gensym(); syms = gensym()
    
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
    
    _lp = gensym()
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

        ($var, $_lp) = if isa($dist, AbstractVector)
            Turing.assume($sampler, $dist, $varname, $var, $vi)
        else
            Turing.assume($sampler, $dist, $varname, $vi)
        end
        $vi.logp += $_lp
    end
end

function tilde(left, right, model_info)
    return generate_observe(left, right, model_info)
end

function tilde(left::Union{Symbol, Expr}, right, model_info)
    if left isa Symbol
        vsym = left
    else
        vsym = getvsym(left)
    end
    @assert isa(vsym, Symbol)
    return _tilde(vsym, left, right, model_info)
end

function _tilde(vsym, left, dist, model_info)
    main_body_names = model_info[:main_body_names]
    model_name = main_body_names[:model]

    if vsym in model_info[:arg_syms]
        if !(vsym in model_info[:dvars_list])
            @debug " Observe - `$(vsym)` is an observation"
            push!(model_info[:dvars_list], vsym)
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
        if !(vsym in model_info[:pvars_list])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @debug msg
            push!(model_info[:pvars_list], vsym)
        end

        return generate_assume(left, dist, model_info)
    end
end

#################
# Main Compiler #
#################

"""
    @model(name, fbody)

Macro to specify a probabilistic model.

Example:

```julia
@model Gaussian(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt.(s))
    end
    return (s, m)
end
```

Compiler design: `sample(fname(x,y), sampler)`.
```julia
fname(x=nothing,y=nothing) = begin
    ex = quote
        # Pour in kwargs for those args where value != nothing.
        fname_model(vi::VarInfo, sampler::Sampler; x = x, y = y) = begin
            vi.logp = zero(Real)
          
            # Pour in model definition.
            x ~ Normal(0,1)
            y ~ Normal(x, 1)
            return x, y
        end
    end
    return Main.eval(ex)
end
```
"""
macro model(input_expr)
    build_model_info(input_expr) |> translate_tilde! |> update_args! |> build_output
end

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
        :dvars_list => Symbol[],
        :pvars_list => Symbol[],
        :main_body_names => Dict(
            :vi => gensym(), 
            :sampler => gensym(),
            :model => gensym(),
            :pvars => gensym(),
            :dvars => gensym(),
            :data => gensym(),
            :closure => gensym()
        )
    )

    return model_info
end

function translate_tilde!(model_info)
    ex = model_info[:main_body]
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, model_info) : x, ex)
    model_info[:main_body] = ex
    return model_info
end

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

function build_output(model_info)
    # Construct user-facing function
    main_body_names = model_info[:main_body_names]
    vi_name = main_body_names[:vi]
    model_name = main_body_names[:model]
    sampler_name = main_body_names[:sampler]
    data_name = main_body_names[:data]
    pvars_name = main_body_names[:pvars]
    dvars_name = main_body_names[:dvars]
    closure_name = main_body_names[:closure]

    args = model_info[:args]
    arg_syms = model_info[:arg_syms]
    outer_function_name = model_info[:name]
    pvars_list = model_info[:pvars_list]
    dvars_list = model_info[:dvars_list]
    closure_main_body = model_info[:main_body]
    arg_defaults = model_info[:arg_defaults]

    if length(dvars_list) == 0
        dvars_nt = :(NamedTuple())
    else
        dvars_nt = :($([:($var = $var) for var in dvars_list]...),)
    end

    unwrap_data_expr = Expr(:block)
    for var in dvars_list
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
        $outer_function_name(;$(args...)) = $outer_function_name($(arg_syms...))
        function $outer_function_name($(args...))
            $pvars_name, $dvars_name = Turing.get_vars($(Tuple{pvars_list...}), $(dvars_nt))
            $data_name = Turing.get_data($dvars_name, $dvars_nt)
            
            $closure_name($sampler_name::Turing.AnySampler, $model_name) = $closure_name($model_name)
            $closure_name($model_name) = $closure_name(Turing.VarInfo(), Turing.SampleFromPrior(), $model_name)
            $closure_name($vi_name::Turing.VarInfo, $model_name) = $closure_name($vi_name, Turing.SampleFromPrior(), $model_name)
            function $closure_name($vi_name::Turing.VarInfo, $sampler_name::Turing.AnySampler, $model_name)
                $unwrap_data_expr
                $vi_name.logp = zero(Real)
                $closure_main_body
            end
            $model_name = Turing.Model{$pvars_name, $dvars_name}($closure_name, $data_name)
            return $model_name
        end
    end)
end

@generated function get_vars(pvars::Type{Tpvars}, dvars_nt::NamedTuple) where {Tpvars <: Tuple}
    pvar_syms = [Tpvars.types...]
    dvar_syms = [dvars_nt.names...]
    dvar_types = [dvars_nt.types...]
    append!(pvar_syms, [dvar_syms[i] for i in 1:length(dvar_syms) if dvar_types[i] == Nothing])
    setdiff!(dvar_syms, pvar_syms)    
    pvars_tuple = Tuple{pvar_syms...}
    dvars_tuple = Tuple{dvar_syms...}

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
