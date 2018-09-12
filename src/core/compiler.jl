using Base.Meta: parse

#################
# Overload of ~ #
#################

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

function generate_observe(left, right)
    obsexpr = esc( 
                quote
                    _lp += Turing.observe(
                        sampler,
                        $(right),   # Distribution
                        $(left),    # Data point
                        vi
                    )
                end 
            )
    return obsexpr
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

    # check if left-hand side is a observation
    if left in Turing._compiler_[:args]
        if ~(left in Turing._compiler_[:dvars])
            @info " Observe - `$(left)` is an observation"
            push!(Turing._compiler_[:dvars], left)
        end

        return generate_observe(left, right)
    else
        # assume its a paramter
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
        assume_ex = quote
            vn = Turing.VarName(vi, $syms, "")
            if isa($(right), Vector)
                $(left), __lp = Turing.assume(
                                              sampler,
                                              $(right),   # dist
                                              vn,         # VarName
                                              $(left),
                                              vi          # VarInfo
                                             )
                _lp += __lp
            else
                $(left), __lp = Turing.assume(
                                              sampler,
                                              $(right),   # dist
                                              vn,         # VarName
                                              vi          # VarInfo
                                             )
                _lp += __lp
            end
        end
        # end of quote block
        return(esc(assume_ex))
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
        assume_ex = quote
            sym, idcs, csym = @VarName $left
            csym_str = string(Turing._compiler_[:name])*string(csym)
            indexing = isempty(idcs) ? "" : mapreduce(idx -> string(idx), *, idcs)

            vn = Turing.VarName(vi, Symbol(csym_str), sym, indexing)

            $(left), __lp = Turing.assume(
                                          sampler,
                                          $right,   # dist
                                          vn,       # VarName
                                          vi        # VarInfo
                                         )
            _lp += __lp
        end
        return esc(assume_ex)
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
"""
macro model(fexpr)
    # Compiler design: sample(fname_compiletime(x,y), sampler)
    #   fname_compiletime(x=nothing,y=nothing; data=data,compiler=compiler) = begin
    #      ex = quote
    #          fname_runtime(vi::VarInfo,sampler::Sampler) = begin
    #              x=x,y=y
    #              # pour all variables in data dictionary, e.g.
    #              k = data[:k]
    #              # pour model definition `fbody`, e.g.
    #              x ~ Normal(0,1)
    #              k ~ Normal(x, 1)
    #          end
    #      end
    #      Main.eval(ex)
    #   end

    # translate all ~ occurences to macro calls
    fexpr = translate(fexpr)

    # extract model name (:name), arguments (:args), (:kwargs) and definition (:body)
    modeldef = MacroTools.splitdef(fexpr)

    lines = length(filter(l -> !isa(l, LineNumberNode), modeldef[:body].args))
    if lines < 1 # function body of the model is empty
        @warn("Model definition seems empty, still continue.")
    end

   # # adjust args, i.e. add 
    #fargs_outer = deepcopy(fargs)

    # Add data argument to outer function
    #push!(fargs_outer[1].args, Expr(:kw, :compiler, compiler))

    # turn f(x;..) into f(x=nothing;..)
    #for i = 2:length(fargs_outer)
    #    if isa(fargs_outer[i], Symbol)
    #        fargs_outer[i] = Expr(:kw, fargs_outer[i], :nothing)
    #    end
    #end

    # construct compiler dictionary
    compiler = Dict(
        :name => modeldef[:name],
        :closure_name => Symbol(modeldef[:name], :_model),
        :args => [],
        :kwargs => modeldef[:kwargs],
        :dvars => Set{Symbol}(),
        :pvars => Set{Symbol}()
    )

    # define model body
    body = insertvarinfo(modeldef[:body])

    # manipulate the function arguments
    fargs = deepcopy(vcat(modeldef[:args], modeldef[:kwargs]))
    for i in 1:length(fargs)
        if isa(fargs[i], Symbol)
            fargs[i] = Expr(:kw, fargs[i], :nothing)
        end
    end

    # construct user function
    fdefn = Dict(
        :name => compiler[:name],
        :kwargs => [Expr(:kw, :compiler, compiler)],
        :args => fargs,
        :body => Expr(:return, compiler[:closure_name])
    )

    modelfun = MacroTools.combinedef(fdefn)

    # construct closure
    closure_def = Dict(
        :name => compiler[:closure_name],
        :kwargs => [],
        :args => [
            :(vi::Turing.VarInfo),
            :(sampler::Union{Nothing, Turing.Sampler})
        ],
        :body => body
    )
    closure = MacroTools.combinedef(closure_def)

    # construct aliases
    alias1 = MacroTools.combinedef(
                Dict(
                    :name => compiler[:closure_name],
                    :args => [:(vi::Turing.VarInfo)],
                    :kwargs => [],
                    :body => :(return $(compiler[:closure_name])(vi, nothing))

                )
    )

    alias2 = MacroTools.combinedef(
                Dict(
                    :name => compiler[:closure_name],
                    :args => [:(sampler::Turing.Sampler)],
                    :kwargs => [],
                    :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), nothing))

                )
    )

    alias3 = MacroTools.combinedef(
                Dict(
                    :name => compiler[:closure_name],
                    :args => [],
                    :kwargs => [],
                    :body => :(return $(compiler[:closure_name])(Turing.VarInfo(), nothing))

                )
    )

    # add definitions to the compiler
    compiler[:alias3] = alias3
    compiler[:alias2] = alias2
    compiler[:alias1] = alias1
    compiler[:closure] = closure

    # add function definitions
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias3)))
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias2)))
    pushfirst!(modelfun.args[2].args, :(Main.eval(alias1)))
    pushfirst!(modelfun.args[2].args, :(Main.eval( Expr(:(=), modelname, closure) )))

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
                    @error("Data `"*$_k_str*"` is not provided.")
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

    pushfirst!(modelfun.args[2].args, quote
        Turing.eval(:(_compiler_ = deepcopy($compiler)))

        # Copy the expr of function definition and callbacks
        alias3 = Turing._compiler_[:alias3]
        alias2 = Turing._compiler_[:alias2]
        alias1 = Turing._compiler_[:alias1]
        closure = Turing._compiler_[:closure]
        modelname = Turing._compiler_[:closure_name]
    end)

    return esc(modelfun)
end

###################
# Helper function #
###################
function setkwargs(fexpr::Expr, kw::Symbol, value)

    # split up the function definition
    funcdef = MacroTools.splitdef(fexpr)

    function insertvi(x)
        return @capture(x, return _) ? Expr(:block, :(vi.logp = _lp), x) : x
    end

    expr_new = MacroTools.postwalk(x->insertvi(x), funcdef[:body])

    # add the new keyword argument
    push!(funcdef[:kwargs], Expr(:kw, kw, value))

    # recompose the function
    return MacroTools.combinedef(funcdef)
end

"""
  insertvarinfo(fexpr::Expr)

Insert `_lp=0` to function call and set `vi.logp=_lp` inplace at the end.
"""
insertvarinfo(fexpr::Expr) = insertvarinfo!(deepcopy(fexpr))
function insertvarinfo!(fexpr::Expr)
    pushfirst!(fexpr.args, :(_lp = zero(Real)))

    # check for the existence of a return statement
    found = false
    MacroTools.postwalk(x -> @capture(x, return _) ? found = true : found = found, fexpr)

    fexpr_new = if !found
        push!(fexpr.args, :(vi.logp = _lp))
        fexpr
    else
        # traverse expression tree and insert :(vi.logp = _lp) before return statements
        function insertvi(x)
            return @capture(x, return _) ? Expr(:block, :(vi.logp = _lp), x) : x
        end
        MacroTools.postwalk(x->insertvi(x), fexpr)
    end

    @info fexpr_new

    return fexpr_new
end

function insdelim(c, deli=",")
    return reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end

getvsym(s::Symbol) = s
getvsym(expr::Expr) = begin
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    curr = expr
    while isa(curr, Expr) && curr.head == :ref
        curr = curr.args[1]
    end
    return curr
end


translate!(ex::Any) = ex
translate!(ex::Expr) = begin
    if (ex.head === :call && ex.args[1] === :(~))
        ex.head = :macrocall; ex.args[1] = Symbol("@~")
        insert!(ex.args, 2, LineNumberNode(-1)) # NOTE: a `LineNumberNode` object is required
        #       at the second args for macro call in 0.7
    else
        map(translate!, ex.args)
    end
    return ex
end
translate(ex::Expr) = translate!(deepcopy(ex))
