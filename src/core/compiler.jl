using Base.Meta: parse

#################
# Overload of ~ #
#################

struct CallableModel{F}
    f::F
    pvars::Set{Symbol}
    dvars::Set{Symbol}
end
(m::CallableModel)(args...; kwargs...) = m.f(args...; kwargs...)

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
    generate_observe(observation, distribution)

Generate an observe expression for observation `observation` drawn from 
a distribution or a vector of distributions (`distribution`).
"""
function generate_observe(observation, distribution)
    return quote
            isdist = if isa($(distribution), AbstractVector)
                # Check if the right-hand side is a vector of distributions.
                all(d -> isa(d, Distribution), $(distribution))
            else
                # Check if the right-hand side is a distribution.
                isa($(distribution), Distribution)
            end
            @assert isdist @error($(wrong_dist_errormsg(@__LINE__)))

            vi.logp += Turing.observe(
                sampler,
                $(distribution),
                $(observation),
                vi
            )
        end
end

"""
    generate_assume(variable, distribution, model_info)

Generate an assume expression for parameters `variable` drawn from 
a distribution or a vector of distributions (`distribution`).

"""
function generate_assume(variable::Union{Symbol, Expr}, distribution, model_info)
    if variable isa Symbol
        sym, idcs, csym = @VarName(variable)
        csym = Symbol(model_info[:name], csym)
        syms = Symbol[csym, variable]
        varname_expr = :(varname = Turing.VarName(vi, $syms, ""))
    else
        sym, idcs, csym = @VarName variable
        csym_str = string(model_info[:name])*string(csym)
        indexing = mapfoldl(string, *, idcs, init = "")
        varname_expr = :(varname = Turing.VarName(vi, Symbol($csym_str), $sym, $indexing))
    end
    return quote
        $varname_expr
        isdist = if isa($(distribution), AbstractVector)
            # Check if the right-hand side is a vector of distributions.
            all(d -> isa(d, Distribution), $(distribution))
        else
            # Check if the right-hand side is a distribution.
            isa($(distribution), Distribution)
        end
        @assert isdist @error($(wrong_dist_errormsg(@__LINE__)))

        ($(variable), _lp) = if isa($(distribution), AbstractVector)
            Turing.assume(
                sampler,
                $(distribution),
                varname,
                $(variable),
                vi
            )
        else
            Turing.assume(
                sampler,
                $(distribution),
                varname,
                vi
            )
        end
        vi.logp += _lp
    end
end

function tilde(left, right, model_info)
    return generate_observe(left, right)
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
    if vsym in model_info[:args]
        if !(vsym in model_info[:dvars])
            @debug " Observe - `$(vsym)` is an observation"
            push!(model_info[:dvars], vsym)
        end

        return quote 
            if $vsym == nothing
                $(generate_assume(left, dist, model_info))
            else
                $(generate_observe(left, dist))
            end
        end
    else
        # Assume it is a parameter.
        if !(vsym in model_info[:pvars])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @debug msg
            push!(model_info[:pvars], vsym)
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
macro model(fexpr)
    # Extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(fexpr)
    # Function body of the model is empty
    warn_empty(modeldef[:body])
    # Construct model_info dictionary
    model_info = Dict(
        :name => modeldef[:name],
        :closure_name => gensym(),
        :args => modeldef[:args],
        :kwargs => modeldef[:kwargs],
        :dvars => Set{Symbol}(),
        :pvars => Set{Symbol}()
    )
    # Unwrap ~ expressions and extract dvars and pvars into `model_info`
    fexpr = translate(fexpr, model_info)

    fargs = modeldef[:args]
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        end
    end    

    # Construct user-facing function
    outer_function_name = model_info[:name]
    pvars = model_info[:pvars]
    dvars = model_info[:dvars]

    closure_name = model_info[:closure_name]
    # Updated body after expanding ~ expressions
    closure_main_body = MacroTools.splitdef(fexpr)[:body]

    return esc(quote
        function $(outer_function_name)($(fargs...))
            $closure_name(sampler::Turing.AnySampler) = $closure_name()
            $closure_name() = $closure_name(Turing.VarInfo(), Turing.SampleFromPrior())
            $closure_name(vi::Turing.VarInfo) = $closure_name(vi, Turing.SampleFromPrior())
            function $closure_name(vi::Turing.VarInfo, sampler::Turing.AnySampler)
                vi.logp = zero(Real)
                $closure_main_body
            end
            return Turing.CallableModel($closure_name, $pvars, $dvars)
        end
    end)
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

translate!(ex::Any, model_info) = ex
function translate!(ex::Expr, model_info)
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, model_info) : x, ex)
    return ex
end
translate(ex::Expr, model_info) = translate!(deepcopy(ex), model_info)
