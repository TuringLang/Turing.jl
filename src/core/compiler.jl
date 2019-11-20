macro varinfo()
    :(throw(_error_msg()))
end
macro logpdf()
    :(throw(_error_msg()))
end
_error_msg() = "This macro is only for use in the `Turing.@model` macro and not for external use."

"""
Usage: @varname x[1,2][1+5][45][3]
  return: VarName{:x}("[1,2][6][45][3]")
"""
macro varname(expr::Union{Expr, Symbol})
    expr |> varname |> esc
end
function varname(expr)
    ex = deepcopy(expr)
    (ex isa Symbol) && return quote
        Turing.VarName{$(QuoteNode(ex))}("")
    end
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    inds = :(())
    while ex.head == :ref
        if length(ex.args) >= 2
            strs = map(x -> :(string($x)), ex.args[2:end])
            pushfirst!(inds.args, :("[" * join($(Expr(:vect, strs...)), ",") * "]"))
        end
        ex = ex.args[1]
        isa(ex, Symbol) && return quote
            Turing.VarName{$(QuoteNode(ex))}(foldl(*, $inds, init = ""))
        end
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end

macro vsym(expr::Union{Expr, Symbol})
    expr |> vsym
end
function vsym(expr::Union{Expr, Symbol})
    ex = deepcopy(expr)
    (ex isa Symbol) && return QuoteNode(ex)
    (ex.head == :ref) || throw("VarName: Mis-formed variable name $(expr)!")
    while ex.head == :ref
        ex = ex.args[1]
        isa(ex, Symbol) && return QuoteNode(ex)
    end
    throw("VarName: Mis-formed variable name $(expr)!")
end

function split_var_str(var_str, inds_as = Vector)
    ind = findfirst(c -> c == '[', var_str)
    if inds_as === String
        if ind === nothing
            return var_str, ""
        else
            return var_str[1:ind-1], var_str[ind:end]
        end
    end
    @assert inds_as === Vector
    inds = Vector{String}[]
    if ind === nothing
        return var_str, inds
    end
    sym = var_str[1:ind-1]
    ind = length(sym)
    while ind < length(var_str)
        ind += 1
        @assert var_str[ind] == '['
        push!(inds, String[])
        while var_str[ind] != ']'
            ind += 1
            if var_str[ind] == '['
                ind2 = findnext(c -> c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2]))
                ind = ind2+1
            else
                ind2 = findnext(c -> c == ',' || c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2-1]))
                ind = ind2
            end
        end
    end
    return sym, inds
end

# Check if the right-hand side is a distribution.
function assert_dist(dist; msg)
    isa(dist, Distribution) || throw(ArgumentError(msg))
end
function assert_dist(dist::AbstractVector; msg)
    all(d -> isa(d, Distribution), dist) || throw(ArgumentError(msg))
end

function wrong_dist_errormsg(l)
    return "Right-hand side of a ~ must be subtype of Distribution or a vector of " *
        "Distributions on line $(l)."
end

macro preprocess(data_vars, missing_vars, ex)
    ex
end
macro preprocess(data_vars, missing_vars, ex::Union{Symbol, Expr})
    sym = gensym(:sym)
    lhs = gensym(:lhs)
    return esc(quote
        # Extract symbol
        $sym = Val($(vsym(ex)))
        # This branch should compile nicely in all cases except for partial missing data
        # For example, when `ex` is `x[i]` and `x isa Vector{Union{Missing, Float64}}`
        if !Turing.Core.inparams($sym, $data_vars) || Turing.Core.inparams($sym, $missing_vars)
            $(varname(ex))
        else
            if Turing.Core.inparams($sym, $data_vars)
                # Evaluate the lhs
                $lhs = $ex
                if ismissing($lhs)
                    $(varname(ex))
                else
                    $lhs
                end
            else
                throw("This point should not be reached. Please report this error.")
            end
        end
    end)
end
@generated function inparams(::Val{s}, ::Val{t}) where {s, t}
    return (s in t) ? :(true) : :(false)
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
model_generator(; x, y)) = model_generator(x, y)
function model_generator(x, y)
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, model)
        if model.args.x isa Type && (model.args.x <: AbstractFloat || model.args.x <: AbstractArray)
            x = Turing.Core.get_matching_type(sampler, vi, model.args.x)
        else
            x = model.args.x
        end
        if model.args.y isa Type && (model.args.y <: AbstractFloat || model.args.y <: AbstractArray)
            y = Turing.Core.get_matching_type(sampler, vi, model.args.y)
        else
            y = model.args.y
        end

        vi.logp = 0
        ...
    end
    return Turing.Model(inner_function, (x = x, y = y))
end
```

Generating a model: `model_generator(x_value)::Model`.
"""
macro model(input_expr)
    build_model_info(input_expr) |> replace_tilde! |> replace_vi! |> 
        replace_logpdf! |> build_output
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

    # Extracting the argument symbols from the model definition
    arg_syms = map(modeldef[:args]) do arg
        # @model demo(x)
        if (arg isa Symbol)
            arg
        # @model demo(::Type{T}) where {T}
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            T
        # @model demo(x = 1)
        elseif MacroTools.@capture(arg, x_ = val_)
            x
        else
            throw(ArgumentError("Unsupported argument $arg to the `@model` macro."))
        end
    end
    if length(arg_syms) == 0
        args_nt = :(NamedTuple())
    else
        nt_type = Expr(:curly, :NamedTuple, 
            Expr(:tuple, QuoteNode.(arg_syms)...), 
            Expr(:curly, :Tuple, [:(Turing.Core.get_type($x)) for x in arg_syms]...)
        )
        args_nt = Expr(:call, :(Turing.namedtuple), nt_type, Expr(:tuple, arg_syms...))
    end
    args = map(modeldef[:args]) do arg
        if (arg isa Symbol)
            arg
        elseif MacroTools.@capture(arg, ::Type{T_} = Tval_)
            if in(T, modeldef[:whereparams])
                S = :Any
            else
                ind = findfirst(x -> MacroTools.@capture(x, T1_ <: S_) && T1 == T, modeldef[:whereparams])
                ind != nothing || throw(ArgumentError("Please make sure type parameters are properly used. Every `Type{T}` argument need to have `T` in the a `where` clause"))
            end
            Expr(:kw, :($T::Type{<:$S}), Tval)
        else
            arg
        end
    end
    model_info = Dict(
        :name => modeldef[:name],
        :main_body => modeldef[:body],
        :arg_syms => arg_syms,
        :args_nt => args_nt,
        :args => args,
        :whereparams => modeldef[:whereparams],
        :main_body_names => Dict(
            :ctx => gensym(:ctx),
            :vi => gensym(:vi),
            :sampler => gensym(:sampler),
            :model => gensym(:model),
            :inner_function => gensym(:inner_function),
            :defaults => gensym(:defaults)
        )
    )

    return model_info
end

"""
    replace_vi!(model_info)

Replaces @varinfo() expressions to the VarInfo instance.
"""
function replace_vi!(model_info)
    ex = model_info[:main_body]
    vi = model_info[:main_body_names][:vi]
    ex = MacroTools.postwalk(x -> @capture(x, @varinfo()) ? vi : x, ex)
    model_info[:main_body] = ex
    return model_info
end

"""
    replace_logpdf!(model_info)

Replaces @logpdf() expressions to the VarInfo instance.
"""
function replace_logpdf!(model_info)
    ex = model_info[:main_body]
    vi = model_info[:main_body_names][:vi]
    ex = MacroTools.postwalk(x -> @capture(x, @logpdf()) ? :($vi.logp) : x, ex)
    model_info[:main_body] = ex
    return model_info
end

"""
    replace_tilde!(model_info)

Replaces ~ expressions to observation or assumption expressions, updating `model_info`.
"""
function replace_tilde!(model_info)
    ex = model_info[:main_body]
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, model_info) : x, ex)
    ex = MacroTools.postwalk(x -> @capture(x, L_ .~ R_) ? dot_tilde(L, R, model_info) : x, ex)
    model_info[:main_body] = ex
    return model_info
end

"""
    tilde(left, right, model_info)

The `tilde` function generates observation expression for data variables and assumption expressions for parameter variables, updating `model_info` in the process.
"""
function tilde(left, right, model_info)
    arg_syms = Val((model_info[:arg_syms]...,))
    model = model_info[:main_body_names][:model]
    vi = model_info[:main_body_names][:vi]
    ctx = model_info[:main_body_names][:ctx]
    sampler = model_info[:main_body_names][:sampler]
    out = gensym(:out)
    lp = gensym(:lp)
    assert_ex = :(Turing.Core.assert_dist($right, msg = $(wrong_dist_errormsg(@__LINE__))))
    if left isa Symbol || left isa Expr
        ex = quote
            $assert_ex
            $out = Turing.Inference.tilde($ctx, $sampler, $right, Turing.Core.@preprocess($arg_syms, Turing.getmissing($model), $left), $vi)
            if $out isa Tuple
                $left, $lp = $out
                $vi.logp += $lp
            else
                $vi.logp += $out
            end
        end
    else
        ex = quote
            $assert_ex
            $out = Turing.observe($sampler, $right, $left, $vi)
            $vi.logp += $out
        end
    end
    return ex
end

function dot_tilde(left, right, model_info)
    arg_syms = Val((model_info[:arg_syms]...,))
    model = model_info[:main_body_names][:model]
    vi = model_info[:main_body_names][:vi]
    ctx = model_info[:main_body_names][:ctx]
    sampler = model_info[:main_body_names][:sampler]
    out = gensym(:out)
    temp = gensym(:temp)
    lp = gensym(:lp)
    assert_ex = :(Turing.Core.assert_dist($right, msg = $(wrong_dist_errormsg(@__LINE__))))
    if left isa Symbol || left isa Expr
        ex = quote
            $assert_ex
            $out = Turing.Inference.dot_tilde($ctx, $sampler, $right, $left, Turing.Core.@preprocess($arg_syms, Turing.getmissing($model), $left), $vi)
            if $out isa Tuple
                $temp, $lp = $out
                $left .= $temp
                $vi.logp += $lp
            else
                $vi.logp += $out
            end
        end
    else
        ex = quote
            $assert_ex
            $out = Turing.dot_observe($sampler, $right, $left, $vi)
            $vi.logp += $out
        end
    end
    return ex
end

"""
    build_output(model_info)

Builds the output expression.
"""
function build_output(model_info)
    # Construct user-facing function
    main_body_names = model_info[:main_body_names]
    ctx = main_body_names[:ctx]
    vi = main_body_names[:vi]
    model = main_body_names[:model]
    sampler = main_body_names[:sampler]
    inner_function = main_body_names[:inner_function]

    # Arguments with default values
    args = model_info[:args]
    # Argument symbols without default values
    arg_syms = model_info[:arg_syms]
    # Arguments namedtuple
    args_nt = model_info[:args_nt]
    # Default values of the arguments
    whereparams = model_info[:whereparams]
    # Model generator name
    outer_function = model_info[:name]
    # Main body of the model
    main_body = model_info[:main_body]

    unwrap_data_expr = Expr(:block)
    for var in arg_syms
        temp_var = gensym(:temp_var)
        varT = gensym(:varT)
        push!(unwrap_data_expr.args, quote
            $temp_var = $model.args.$var
            $varT = typeof($temp_var)
            if $temp_var isa Type && ($temp_var <: AbstractFloat || $temp_var <: AbstractArray)
                $var = Turing.Core.get_matching_type($sampler, $vi, $temp_var)
            elseif $varT <: Array && Missing <: eltype($varT)
                $var = Turing.Core.get_matching_type($sampler, $vi, $varT)($temp_var)
            else
                $var = $temp_var
            end
        end)
    end
    return esc(quote
        # Allows passing arguments as kwargs
        $outer_function(;$(args...)) = $outer_function($(arg_syms...))
        function $outer_function($(args...))
            function $inner_function(
                $vi::Turing.VarInfo,
                $sampler::Turing.AbstractSampler,
                $ctx::Turing.AbstractContext,
                $model
                )

                $unwrap_data_expr
                $vi.logp = 0
                $main_body
            end
            return Turing.Model($inner_function, $args_nt)
        end
    end)
end

# A hack for NamedTuple type specialization
# (T = Int,) has type NamedTuple{(:T,), Tuple{DataType}} by default
# With this function, we can make it NamedTuple{(:T,), Tuple{Type{Int}}}
# Both are correct, but the latter is what we want for type stability
get_type(::Type{T}) where {T} = Type{T}
get_type(t) = typeof(t)

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return
end

"""
    get_matching_type(spl, vi, ::Type{T}) where {T}
Get the specialized version of type `T` for sampler `spl`. For example,
if `T === Float64` and `spl::Hamiltonian`, the matching type is `eltype(vi[spl])`.
"""
get_matching_type(spl, vi, ::Type{T}) where {T} = T
