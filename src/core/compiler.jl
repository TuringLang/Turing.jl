using Base.Meta: parse

"""
    @data(model_f(), data_dict)

Manipulate the keyword arguments of the function call of model_f() to use the data 
specified in the data_dict dictionary.

Example:
```julia
f(;x = 1) = 2*x
d = Dict(x => 2)

@data(f(), d) # => 4
```
"""
macro data(fexpr::Expr, dexpr::Expr)
    if (dexpr.args[1] == :Dict)
        dargs = filter(d -> d.args[1] == :(=>), dexpr.args[2:end])
        fname = string(fexpr.args[1])

        fcall = Expr(:call, Symbol(fname))
        @debug(fexpr)

        existingkws = fexpr.args[2:end]
        existingkws_var = Symbol[]

        for kw in existingkws
            push!(existingkws_var, kw.args[1])
            push!(fcall.args, Expr(:kw, kw.args[1], kw.args[2]))
        end

        for kw in dargs
            var = isa(kw.args[2], Symbol) ? kw.args[2] : kw.args[2].value
            @assert isa(var, Symbol)

            if !(var in existingkws_var)
                push!(fcall.args, Expr(:kw, var, kw.args[3]))
            end
        end
        return esc(fcall)
    else
        @warn("Unexpected second argument: ", dexpr)
    end
end

macro data(fexpr::Expr, d::Symbol)

    # evaluate d in the main scope
    ddata = Main.eval(d)
    
    if isa(ddata, Dict)

        fname = string(fexpr.args[1])

        fcall = Expr(:call, Symbol(fname))
        @debug(fexpr)

        existingkws = fexpr.args[2:end]
        existingkws_var = Symbol[]

        for kw in existingkws
            push!(existingkws_var, kw.args[1])
            push!(fcall.args, Expr(:kw, kw.args[1], kw.args[2]))
        end

        for key in keys(ddata)
            if !(key in existingkws_var)
                push!(fcall.args, Expr(:kw, key, ddata[key]))
            end
        end
        esc(fcall)
    else
        @warn("Unexpected data type, got a $(typeof(ddata)) was expecting a Dict!")
    end
end

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
      _2   = _2.args[1]
      isa(_2, Symbol) && (_1 = ":($_2)" * ", ($_1), Symbol(\"$s\")"; break)
    end
    return esc(parse(_1))
  else
    error("VarName: Mis-formed variable name $(ex)!")
  end
end

"""
    var_name ~ Distribution()

Tilde notation `~` can be used to specifiy *a variable follows a distributions*.

If `var_name` is an un-defined variable or a container (e.g. Vector or Matrix), this variable will be treated as model parameter; otherwise if `var_name` is defined, this variable will be treated as data.
"""
macro ~(left::Real, right)

    # Call observe
    esc(
        quote
            _lp += Turing.observe(
                                  sampler,
                                  $(right),   # Distribution
                                  $(left),    # Data point
                                  vi
                                 )
        end
       )
end

macro ~(left, right)
    vsym = getvsym(left)
    vsym_str = string(vsym)
    # Require all data to be stored in data dictionary.
    if vsym in Turing._compiler_[:fargs]
        if ~(vsym in Turing._compiler_[:dvars])
            @info " Observe - `" * vsym_str * "` is an observation"
            push!(Turing._compiler_[:dvars], vsym)
        end
        esc(
            quote
                # Call observe
                _lp += Turing.observe(
                                      sampler,
                                      $(right),   # Distribution
                                      $(left),    # Data point
                                      vi
                                     )

            end
           )
    else
      if ~(vsym in Turing._compiler_[:pvars])
        msg = " Assume - `" * vsym_str * "` is a parameter"
        isdefined(Main, Symbol(vsym_str)) && (msg  *= " (ignoring `$(vsym_str)` found in global scope)")
        @info msg
        push!(Turing._compiler_[:pvars], vsym)
      end
      # The if statement is to deterimnet how to pass the prior.
      # It only supports pure symbol and Array(/Dict) now.
      #csym_str = string(gensym())
      if isa(left, Symbol)
        # Symbol
        sym, idcs, csym = @VarName(left)
        csym = Symbol(string(Turing._compiler_[:fname])*string(csym))
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
      else
        assume_ex = quote
          sym, idcs, csym = @VarName $left
          csym_str = string(Turing._compiler_[:fname])*string(csym)
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
        esc(assume_ex)
        end
    end
end

#################
# Main Compiler #
#################

"""
    @model(name, fbody)

Wrapper for models.

Usage:

```julia
@model model() = begin
# body
end
```

Example:

```julia
@model gauss() = begin
s ~ InverseGamma(2,3)
m ~ Normal(0,sqrt.(s))
1.5 ~ Normal(m, sqrt.(s))
2.0 ~ Normal(m, sqrt.(s))
return(s, m)
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

    # extract components of the model definition
    fname, fargs, fbody = extractcomponents(fexpr)

    # Insert varinfo expressions into the function body 
    fbody_inner = insertvarinfo(fbody)

    # construct new function
    fname_inner = Symbol("$(fname)_model")
    fname_inner_str = string(fname_inner)
    fdefn_inner_func = constructfunc(
                                     fname_inner,
                                     [
                                      :(vi::Turing.VarInfo),
                                      :(sampler::Union{Nothing,Turing.Sampler})
                                     ],
                                     fbody_inner
                                    )
    fdefn_inner = Expr(:(=), fname_inner, fdefn_inner_func) 

    # construct helper functions
    fdefn_inner_callback_1 = parse("$fname_inner_str(vi::Turing.VarInfo)=$fname_inner_str(vi,nothing)")
    fdefn_inner_callback_2 = parse("$fname_inner_str(sampler::Turing.Sampler)=$fname_inner_str(Turing.VarInfo(),nothing)")
    fdefn_inner_callback_3 = parse("$fname_inner_str()=$fname_inner_str(Turing.VarInfo(),nothing)")

    # construct compiler dictionary
    compiler = Dict(:fname => fname,
                    :fargs => fargs,
                    :fbody => fbody,
                    :dvars => Set{Symbol}(),  # data
    :pvars => Set{Symbol}(),  # parameter
    :fdefn_inner => fdefn_inner,
    :fdefn_inner_callback_1 => fdefn_inner_callback_1,
    :fdefn_inner_callback_2 => fdefn_inner_callback_2,
    :fdefn_inner_callback_3 => fdefn_inner_callback_3)

    # Outer function defintion 1: f(x,y) ==> f(x,y;data=Dict())
    fargs_outer = deepcopy(fargs)

    # Add data argument to outer function -> TODO: move to extra macro!
    #push!(fargs_outer[1].args, Expr(:kw, :data, :(Dict{Symbol,Any}())))

    # Add data argument to outer function
    push!(fargs_outer[1].args, Expr(:kw, :compiler, compiler))

    # turn f(x;..) into f(x=nothing;..)
    for i = 2:length(fargs_outer)
        if isa(fargs_outer[i], Symbol)
            fargs_outer[i] = Expr(:kw, fargs_outer[i], :nothing)
        end
    end

    # construct outer call-back function
    fdefn_outer = constructfunc(
                                fname,
                                fargs_outer,
                                Expr(:block, Expr(:return, fname_inner))
    )

    # add function definitions
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback_3)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback_2)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback_1)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner)))

    # TODO: clean up this part! 
    pushfirst!(fdefn_outer.args[2].args, quote
                   # Check fargs, data
                   Turing.eval(:(_compiler_ = deepcopy($compiler)))
                   fargs = Turing._compiler_[:fargs];

                   # Copy the expr of function definition and callbacks
                   fdefn_inner = Turing._compiler_[:fdefn_inner];
                   fdefn_inner_callback_1 = Turing._compiler_[:fdefn_inner_callback_1];
                   fdefn_inner_callback_2 = Turing._compiler_[:fdefn_inner_callback_2];
                   fdefn_inner_callback_3 = Turing._compiler_[:fdefn_inner_callback_3];

                   # Add gensym to function name
                   fname_inner_with_gensym = gensym((fdefn_inner.args[2].args[1].args[1]));

                   # Change the name of inner function definition to the one with gensym()
                   fdefn_inner.args[2].args[1].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_1.args[1].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_1.args[2].args[2].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_2.args[1].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_2.args[2].args[2].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_3.args[1].args[1] = fname_inner_with_gensym
                   fdefn_inner_callback_3.args[2].args[2].args[1] = fname_inner_with_gensym

                   # Copy data dictionary
                   for k in keys(data)
                       if fdefn_inner.args[2].args[2].args[1].head == :line
                           # Preserve comments, useful for debuggers to
                           # correctly locate source code oringin.
                           insert!(fdefn_inner.args[2].args[2].args, 2, Expr(:(=), Symbol(k), data[k]))
                       else
                           insert!(fdefn_inner.args[2].args[2].args, 1, Expr(:(=), Symbol(k), data[k]))
                       end
                   end
                   # @debug fdefn_inner
               end )

    # TODO: extract into a function ?
    for k in fargs
        if isa(k, Symbol)       # f(x,..)
            _k = k
        elseif k.head == :kw    # f(z=1,..)
            _k = k.args[1]
        else
            _k = nothing
        end
        if _k != nothing
            _k_str = string(_k)
            @debug _k_str, " = ", _k
            data_check_ex = quote
                if haskey(data, keytype(data)($_k_str))
                    if nothing != $_k
                        Turing.dwarn(0, " parameter "*$_k_str*" found twice, value in data dictionary will be used.")
                    end
                else
                    data[keytype(data)($_k_str)] = $_k
                    data[keytype(data)($_k_str)] == nothing && @error("Data `"*$_k_str*"` is not provided.")
                end
            end
            pushfirst!(fdefn_outer.args[2].args, data_check_ex)
        end
    end


    pushfirst!(fdefn_outer.args[2].args, quote data = copy(data) end)

    @debug esc(fdefn_outer)
    esc(fdefn_outer)
end



###################
# Helper function #
###################
function constructfunc(fname::Symbol, fargs, fbody)
    fdefn = Expr(:function, 
                 Expr(:call, 
                      fname,
                      fargs...
                     ),
                 deepcopy(fbody)
                )
    return fdefn
end

insertvarinfo(fexpr::Expr) = insertvarinfo!(deepcopy(fexpr))

function insertvarinfo!(fexpr::Expr)
    return_ex = fexpr.args[end] # get last statement of defined model
    if typeof(return_ex) == Symbol
        pop!(fexpr.args)
    elseif return_ex.head == :return || return_ex.head == :tuple
        pop!(fexpr.args)
    else
        @error("Unknown return expression: $(return_ex)")
    end

    pushfirst!(fexpr.args, :(_lp = zero(Real)))
    push!(fexpr.args, :(vi.logp = _lp))
    push!(fexpr.args, Expr(:return, :vi))

    fexpr
end

function extractcomponents_(fnode::LineNumberNode, fexpr::Expr)
    return extractcomponents_(fexpr.args[1], fexpr.args[2])
end

function extractcomponents_(fexpr::Expr, args)
    return (fexpr, args)
end

function extractcomponents(fexpr::Expr)

    fheader, fbody = extractcomponents_(fexpr.args[1], fexpr.args[2])  

    # model name
    fname = fheader.args[1]
    # model parameters, e.g. (x,y; z= ....)	
    fargs = fheader.args[2:end]

    # Ensure we have allow for keyword arguments, e.g.
    #   f(x,y)
    #       ==> f(x,y;)
    #   f(x,y; c=1)
    #       ==> unchanged

    if (length(fargs) == 0 ||         # e.g. f()
        isa(fargs[1], Symbol) ||  		# e.g. f(x,y)
        fargs[1].head == :kw)     		# e.g. f(x,y=1)
        insert!(fargs, 1, Expr(:parameters))
    end

    return (fname, fargs, fbody)
end


function insdelim(c, deli=",")
    reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end

getvsym(s::Symbol) = s
getvsym(expr::Expr) = begin
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    curr = expr
    while isa(curr, Expr) && curr.head == :ref
        curr = curr.args[1]
    end
    curr
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
    ex
end
translate(ex::Expr) = translate!(deepcopy(ex))
