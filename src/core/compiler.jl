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
    generate_assume(variable, distribution, compiler)

Generate an assume expression for parameters `variable` drawn from 
a distribution or a vector of distributions (`distribution`).

"""
function generate_assume(variable, distribution, compiler)
    sym, idcs, csym = @VarName(variable)
    csym = Symbol(compiler[:name], csym)
    syms = Symbol[csym, variable]
    return quote
            varname = Turing.VarName(vi, $syms, "")

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

function generate_assume(variable::Expr, distribution, compiler)
    sym, idcs, csym = @VarName variable
    csym_str = string(compiler[:name])*string(csym)
    indexing = mapfoldl(string, *, idcs, init = "")
    return quote
            varname = Turing.VarName(vi, Symbol($csym_str), $sym, $indexing)
            # Sanity check.
            isdist = if isa($(distribution), Vector)
                all(d -> isa(d, Distribution), $(distribution))
            else
                isa($(distribution), Distribution)
            end
            @assert isdist @error($(wrong_dist_errormsg(@__LINE__)))

            $(variable), _lp = Turing.assume(
                sampler,
                $(distribution),
                varname,
                vi
            )
            vi.logp += _lp
        end
end

"""
    macro: @~ var Distribution()

Tilde notation macro. This macro constructs Turing.observe or
Turing.assume calls depending on the left-hand argument.
Note that the macro is interconnected with the @model macro and
assumes that a `compiler` struct is available.

Example:
```julia
@~ x Normal()
```
"""
#macro ~(left, right)
#    return tilde(left, right)
#end

function tilde(left::Number, right, compiler)
    generate_observe(left, right)
end

function tilde(left, right, compiler)
    return quote
        println($left)
        if $left isa Nothing
            $(generate_assume(left, right, compiler))
        else
            $(generate_observe(left, right))
        end
    end
end

function tilde(left::Symbol, right, compiler)
    # Check if left-hand side is a observation.
    if left in compiler[:args]
        if !(left in compiler[:dvars])
            @debug " Observe - `$(left)` is an observation"
            push!(compiler[:dvars], left)
        end

        return quote 
            if $left isa Nothing
                $(generate_assume(left, right, compiler))
            else
                $(generate_observe(left, right))
            end
        end
    else
        # Assume it is a parameter.
        if !(left in compiler[:pvars])
            msg = " Assume - `$(left)` is a parameter"
            if isdefined(Main, left)
                msg  *= " (ignoring `$(left)` found in global scope)"
            end

            @debug msg
            push!(compiler[:pvars], left)
        end

        return generate_assume(left, right, compiler)
    end
end

function tilde(left::Expr, right, compiler)
    vsym = getvsym(left)
    @assert isa(vsym, Symbol)

    if vsym in compiler[:args]
        if !(vsym in compiler[:dvars])
            @debug " Observe - `$(vsym)` is an observation"
            push!(compiler[:dvars], vsym)
        end

        return quote 
            if $vsym isa Nothing
                $(generate_assume(left, right, compiler))
            else
                $(generate_observe(left, right))
            end
        end
    else
        if !(vsym in compiler[:pvars])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @debug msg
            push!(compiler[:pvars], vsym)
        end

        return generate_assume(left, right, compiler)
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
fname(x=nothing,y=nothing; compiler=compiler) = begin
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
    # extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(fexpr)
    # function body of the model is empty
    warn_empty(modeldef[:body])
    # construct compiler dictionary
    compiler = Dict(
        :name => modeldef[:name],
        :closure_name => Symbol(modeldef[:name], :_model),
        :args => modeldef[:args],
        :kwargs => modeldef[:kwargs],
        :dvars => Set{Symbol}(),
        :pvars => Set{Symbol}()
    )
    # extract dvars, pvars and unwrap ~ expressions
    fexpr = translate(fexpr, compiler)
    modeldef = MacroTools.splitdef(fexpr)
    fargs = modeldef[:args]
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        end
    end    
    
    # Construct closure.
    closure = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :kwargs => [],
            :args => [
                :(vi::Turing.VarInfo),
                :(sampler::Turing.AnySampler)
            ],
            # Initialise logp in VarInfo.
            :body => Expr(:block, :(vi.logp = zero(Real)), modeldef[:body].args...)
        )
    )

    # Construct aliases.
    alias1 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [:(vi::Turing.VarInfo)],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(vi, Turing.SampleFromPrior()))
        )
    )

    alias2 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [:(sampler::Turing.AnySampler)],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), Turing.SampleFromPrior()))
        )
    )

    alias3 = MacroTools.combinedef(
        Dict(
            :name => compiler[:closure_name],
            :args => [],
            :kwargs => [],
            :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), Turing.SampleFromPrior()))
        )
    )

    # Add definitions to the compiler.
    compiler[:closure] = closure
    compiler[:alias1] = alias1
    compiler[:alias2] = alias2
    compiler[:alias3] = alias3

    # Construct user function.
    modelfun = MacroTools.combinedef(
        Dict(
            :name => compiler[:name],
            :args => fargs,
            :kwargs => [],
            :body => Expr(:block, 
                            # Eval the closure's methods globally and return it
                            closure,
                            alias1,
                            alias2,
                            alias3,
                            :(return Turing.CallableModel($(compiler[:closure_name]), $(compiler[:pvars]), $(compiler[:dvars])))
                        )
        )
    )
    
    return esc(modelfun)
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

translate!(ex::Any, compiler) = ex
function translate!(ex::Expr, compiler)
    ex = MacroTools.postwalk(x -> @capture(x, L_ ~ R_) ? tilde(L, R, compiler) : x, ex)
    return ex
end
translate(ex::Expr, compiler) = translate!(deepcopy(ex), compiler)
