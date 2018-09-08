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
    end
end

function generate_observe(right, left)
    obsexpr = esc( quote
            _lp += Turing.observe(
                                  sampler,
                                  $(right),   # Distribution
                                  $(left),    # Data point
                                  vi
                                 )
        end )
    return obsexpr
end


"""
    var_name ~ Distribution()

Tilde notation `~` can be used to specifiy *a variable follows a distributions*.

If `var_name` is an un-defined variable or a container (e.g. Vector or Matrix), this variable will be treated as model parameter; otherwise if `var_name` is defined, this variable will be treated as data.
"""
macro ~(left::Real, right)
    @info "Observe constant - ", left
    return generate_observe(right, left)
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

        return generate_observe(right, left)
    else
        if ~(vsym in Turing._compiler_[:pvars])
            msg = " Assume - `" * vsym_str * "` is a parameter"
            if isdefined(Main, Symbol(vsym_str))
                msg  *= " (ignoring `$(vsym_str)` found in global scope)"
            end
        
            @info msg
            push!(Turing._compiler_[:pvars], vsym)
        end
      
        # The if statement is to deterimnet how to pass the prior.
        # It only supports pure symbol and Array(/Dict) now.
        #csym_str = string(gensym())
        
        assume_ex = if isa(left, Symbol)
            
            # Symbol
            sym, idcs, csym = @VarName(left)
            csym = Symbol(string(Turing._compiler_[:fname])*string(csym))
            syms = Symbol[csym, left]
            assume_ex_ = quote
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
            assume_ex_
        else
            assume_ex_ = quote
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
            assume_ex_
        end
        esc(assume_ex)
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

    lines = length(filter(l -> !isa(l, LineNumberNode), fbody.args))
    if lines < 1 # function body of the model is empty
        @warn("Model definition seems empty, still continue.")
    end
    
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
    fdefn_inner_callback1 = constructfunc(
                                           fname_inner,
                                           [:(vi::Turing.VarInfo)],
                                           :($fname_inner(vi, nothing))
                                          )
    
    fdefn_inner_callback2 = constructfunc(
                                           fname_inner,
                                           [:(sampler::Turing.Sampler)],
                                           :($fname_inner(Turing.VarInfo(), nothing))
                                          )
    
    fdefn_inner_callback3 = constructfunc(
                                           fname_inner,
                                           [],
                                           :($fname_inner(Turing.VarInfo(), nothing))
                                          )
    # construct compiler dictionary
    compiler = Dict(:fname => fname,
                    :fargs => fargs,
                    :fbody => fbody,
                    :dvars => Set{Symbol}(),  # data
    :pvars => Set{Symbol}(),  # parameter
    :fdefn_inner => fdefn_inner,
    :fdefn_inner_callback_1 => fdefn_inner_callback1,
    :fdefn_inner_callback_2 => fdefn_inner_callback2,
    :fdefn_inner_callback_3 => fdefn_inner_callback3)

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
    @info(fdefn_inner.args[2].args[1].args[1])
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback3)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback2)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback1)))
    pushfirst!(fdefn_outer.args[2].args, :( @info("here"); Main.eval(fdefn_inner)))

    # TODO: clean up this part! 
    pushfirst!(fdefn_outer.args[2].args, quote
                 
                 Turing.eval(:(_compiler_ = deepcopy($compiler)))
                 fargs = Turing._compiler_[:fargs];

                   # Copy the expr of function definition and callbacks
                   fdefn_inner = Turing._compiler_[:fdefn_inner];
                   fdefn_inner_callback1 = Turing._compiler_[:fdefn_inner_callback_1];
                   fdefn_inner_callback2 = Turing._compiler_[:fdefn_inner_callback_2];
                   fdefn_inner_callback3 = Turing._compiler_[:fdefn_inner_callback_3];

                    # Add gensym to function name
                    fname, _, _ = extractcomponents(fdefn_inner)
                    fname_gensym = gensym(fname)

                    # Change the name of inner function definition to the one with gensym()
                    setfname!(fdefn_inner, fname_gensym)
                    
                    setfname!(fdefn_inner_callback1, fname_gensym)
                    setfcall!(fdefn_inner_callback1, fname_gensym)
                    
                    setfname!(fdefn_inner_callback2, fname_gensym)
                    setfcall!(fdefn_inner_callback2, fname_gensym)
                    
                    setfname!(fdefn_inner_callback3, fname_gensym)
                    setfcall!(fdefn_inner_callback3, fname_gensym)

                    @info(fdefn_inner_callback1)
                    @info(fdefn_inner_callback2)
                    @info(fdefn_inner_callback3)

                    @info fdefn_inner
               end )

    esc(fdefn_outer)
end



###################
# Helper function #
###################
function setfcall!_(fexpr::Expr, name::Symbol)
    if fexpr.head == :function
        @assert fexpr.args[2].head == :call
        fexpr.args[2].args[1] = name
    else
        @assert length(fexpr.args) > 1
        setfcall!_(fexpr.args[2], name)
    end
end
function setfcall!(fexpr::Expr, name::Symbol)
    setfcall!_(fexpr, name)
end
function setfname!_(fexpr::Expr, name::Symbol)
    if fexpr.head == :function
        fexpr.args[1].args[1] = name
    else
        @assert length(fexpr.args) > 1
        setfname!_(fexpr.args[2], name)
    end
end
function setfname!(fexpr::Expr, name::Symbol)
    setfname!_(fexpr, name)
end

"""
  constructfunc(fname::Symbol, fargs, fbody)

Construct a function.
"""
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

"""
  insertvarinfo(fexpr::Expr)

Insert `_lp=0` to function call and set `vi.logp=_lp` inplace at the end.
"""
insertvarinfo(fexpr::Expr) = insertvarinfo!(deepcopy(fexpr))
function insertvarinfo!(fexpr::Expr)
	pushfirst!(fexpr.args, :(_lp = zero(Real)))
	return_ex = fexpr.args[end] # get last statement of defined model
    if (typeof(return_ex) == Symbol 
        || return_ex.head == :return
        || return_ex.head == :tuple)

        pop!(fexpr.args)
        push!(fexpr.args, :(vi.logp = _lp))
  	    push!(fexpr.args, return_ex)
	else
  	    push!(fexpr.args, :(vi.logp = _lp))
 	end 
    return fexpr
end

"""
    extractcomponents_(fnode::LineNumberNode, fexpr::Expr)

Internal procedure to extract function header and body.
"""
function extractcomponents_(fnode::LineNumberNode, fexpr::Expr)
    return extractcomponents_(fexpr.args[1], fexpr.args[2])
end

function extractcomponents_(fexpr::Expr, args)
    return (fexpr, args)
end

function extractcomponents_(fnode::Symbol, fexpr::Expr)
    return extractcomponents_(fexpr.args[1], fexpr.args[2])
end
"""
    extractcomponents(fexpr::Expr)

Extract function name, arguments and body.
"""
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
