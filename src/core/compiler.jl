using Base.Meta: parse

#################
# Overload of ~ #
#################

# TODO: Replace this macro, see issue #514
macro VarName(ex::Union{Expr, Symbol})
  # Usage: @VarName x[1,2][1+5][45][3]
  #    return: (:x,[1,2],6,45,3)
    s = string(gensym())
    if isa(ex, Symbol)
        ex_str = string(ex)
        return :(Symbol($ex_str), (), Symbol($s))
    elseif ex.head == :ref
        _2 = ex
        _1 = ""
        while _2.head == :ref
            if length(_2.args) > 2
                _1 = "[" * foldl( (x,y)-> "$x, $y", map(string, _2.args[2:end])) * "], $_1"
            else
                _1 = "[" * string(_2.args[2]) * "], $_1"
            end
            _2 = _2.args[1]
            isa(_2, Symbol) && (_1 = ":($_2)" * ", ($_1), Symbol(\"$s\")"; break)
        end
        return esc(parse(_1))
    else
        @error "VarName: Mis-formed variable name $(ex)!"
        return :()
    end
end


"""
    generate_observe(observation, distribution)

Generate an observe expression.
"""
function generate_observe(observation, distribution)
    return esc(
        quote
            vi.logp += Turing.observe(
                sampler,
                $(distribution),
                $(observation),
                vi
            )
        end
    )
end

"""
    generate_assume(variable, distribution, syms)

Generate an assume expression.
"""
function generate_assume(variable, distribution, syms)
    return esc(
        quote
            varname = Turing.VarName(vi, $syms, "")
            ($(variable), _lp) = if isa($(distribution), Vector)
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
    )
end

function generate_assume(variable::Expr, distribution)
    return esc(
        quote
            sym, idcs, csym = @VarName $variable
            csym_str = string(Turing._compiler_[:name])*string(csym)
            indexing = isempty(idcs) ? "" : mapreduce(idx -> string(idx), *, idcs)
            varname = Turing.VarName(vi, Symbol(csym_str), sym, indexing)

            $(variable), _lp = Turing.assume(
                sampler,
                $(distribution),
                varname,
                vi
            )
            vi.logp += _lp
        end
    )
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
macro ~(left, right)
    return generate_observe(left, right)
end

macro ~(left::Symbol, right)
    # Check if left-hand side is a observation.
    if left in Turing._compiler_[:args]
        if ~(left in Turing._compiler_[:dvars])
            @info " Observe - `$(left)` is an observation"
            push!(Turing._compiler_[:dvars], left)
        end

        return generate_observe(left, right)
    else
        # Assume it is a parameter.
        if ~(left in Turing._compiler_[:pvars])
            msg = " Assume - `$(left)` is a parameter"
            if isdefined(Main, left)
                msg  *= " (ignoring `$(left)` found in global scope)"
            end

            @info msg
            push!(Turing._compiler_[:pvars], left)
        end

        sym, idcs, csym = @VarName(left)
        csym = Symbol(string(Turing._compiler_[:name])*string(csym))
        syms = Symbol[csym, left]

        return generate_assume(left, right, syms)
    end
end

macro ~(left::Expr, right)
    vsym = getvsym(left)
    @assert isa(vsym, Symbol)

    if vsym in Turing._compiler_[:args]
        if ~(vsym in Turing._compiler_[:dvars])
            @info " Observe - `$(vsym)` is an observation"
            push!(Turing._compiler_[:dvars], vsym)
        end

        return generate_observe(left, right)
    else
        if ~(vsym in Turing._compiler_[:pvars])
            msg = " Assume - `$(vsym)` is a parameter"
            if isdefined(Main, vsym)
                msg  *= " (ignoring `$(vsym)` found in global scope)"
            end

            @info msg
            push!(Turing._compiler_[:pvars], vsym)
        end

        return generate_assume(left, right)
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

    # translate all ~ occurences to macro calls
    fexpr = translate(fexpr)

    # extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(fexpr)

    lines = length(filter(l -> !isa(l, LineNumberNode), modeldef[:body].args))
    if lines < 1 # function body of the model is empty
        @warn("Model definition seems empty, still continue.")
    end

    # construct compiler dictionary
    compiler = Dict(
        :name => modeldef[:name],
        :closure_name => Symbol(modeldef[:name], :_model),
        :args => [],
        :kwargs => modeldef[:kwargs],
        :dvars => Set{Symbol}(),
        :pvars => Set{Symbol}()
    )

    # Initialise logp in VarInfo.
    body = modeldef[:body]
    pushfirst!(body.args, :(vi.logp = zero(Real)))

    # Manipulate the function arguments.
    fargs = deepcopy(vcat(modeldef[:args], modeldef[:kwargs]))
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        end
    end

    # Construct user function.
    fdefn = Dict(
        :name => compiler[:name],
        :kwargs => [Expr(:kw, :compiler, compiler)],
        :args => fargs,
        :body => Expr(:return, compiler[:closure_name])
    )

    modelfun = MacroTools.combinedef(fdefn)

    # Construct closure.
    closure_def = Dict(
        :name => compiler[:closure_name],
        :kwargs => [],
        :args => [
            :(vi::Turing.VarInfo),
            :(sampler::Turing.AnySampler)
        ],
        :body => body
    )
    closure = MacroTools.combinedef(closure_def)

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
    compiler[:alias3] = alias3
    compiler[:alias2] = alias2
    compiler[:alias1] = alias1
    compiler[:closure] = closure

    # Add function definitions.
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias3)))
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias2)))
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias1)))
    pushfirst!(modelfun.args[2].args, :(Main.eval( Expr(:(=), modelname, closure) )))

    # Insert argument values as kwargs to the closure.
    for k in fargs
        if isa(k, Symbol)
            _k = k
        elseif k.head == :kw
            _k = k.args[1]
        else
            _k = nothing
        end

        if _k != nothing
            _k_str = string(_k)
            data_insertion = quote
                if $_k == nothing
                    # Notify the user if an argument is missing.
                    @warn("Data `"*$_k_str*"` not provided, treating as parameter instead.")
                else
                    if Symbol($_k_str) ∉ Turing._compiler_[:args]
                        push!(Turing._compiler_[:args], Symbol($_k_str))
                    end
                    closure = Turing.setkwargs(closure, Symbol($_k_str), $_k)
                end
            end
            pushfirst!(modelfun.args[2].args, data_insertion)
        end
    end

    pushfirst!(
        modelfun.args[2].args,  # Body of modelfun.
        quote
            Turing.eval(:(_compiler_ = deepcopy($compiler)))

            # Copy the expr of function definition and callbacks.
            alias3 = Turing._compiler_[:alias3]
            alias2 = Turing._compiler_[:alias2]
            alias1 = Turing._compiler_[:alias1]
            closure = Turing._compiler_[:closure]
            modelname = Turing._compiler_[:closure_name]
        end
    )

    return esc(modelfun)
end


####################
# Helper functions #
####################

function setkwargs(fexpr::Expr, kw::Symbol, value)

    # Split up the function definition.
    funcdef = MacroTools.splitdef(fexpr)

    function insertvi(x)
        return @capture(x, return _) ? Expr(:block, :(vi.logp = _lp), x) : x
    end

    expr_new = MacroTools.postwalk(x->insertvi(x), funcdef[:body])

    # Add the new keyword argument.
    push!(funcdef[:kwargs], Expr(:kw, kw, value))

    # Recompose the function.
    return MacroTools.combinedef(funcdef)
end

getvsym(s::Symbol) = s
function getvsym(expr::Expr)
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    curr = expr
    while isa(curr, Expr) && curr.head == :ref
        curr = curr.args[1]
    end
    return curr
end

translate!(ex::Any) = ex
function translate!(ex::Expr)
    if ex.head === :call && ex.args[1] === :(~)
        ex.head = :macrocall
        ex.args[1] = Symbol("@~")
        insert!(ex.args, 2, LineNumberNode(@__LINE__))
    else
        map(translate!, ex.args)
    end
    return ex
end
translate(ex::Expr) = translate!(deepcopy(ex))
