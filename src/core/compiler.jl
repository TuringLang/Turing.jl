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
    if left in Turing._compiler_[:fargs]
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
        # end of quote block
        return(esc(assume_ex))
    end
end

macro ~(left::Expr, right)
    vsym = getvsym(left)
    @assert isa(vsym, Symbol)
    
    if vsym in Turing._compiler_[:fargs]
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

    # Outer function defintion
    fargs_outer = deepcopy(fargs)

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
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback3)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback2)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner_callback1)))
    pushfirst!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner)))

    # TODO: clean up this part! 
    pushfirst!(fdefn_outer.args[2].args, quote
                   
                    Turing.eval(:(_compiler_ = deepcopy($compiler)))
                    fargs = Turing._compiler_[:fargs];

                    # Copy the expr of function definition and callbacks
                    #fdefn_inner = Turing._compiler_[:fdefn_inner];
                    fdefn_inner_callback1 = Turing._compiler_[:fdefn_inner_callback_1];
                    fdefn_inner_callback2 = Turing._compiler_[:fdefn_inner_callback_2];
                    fdefn_inner_callback3 = Turing._compiler_[:fdefn_inner_callback_3];

                    # Add gensym to function name
                    fname, _, _ = Turing.extractcomponents(fdefn_inner)
                    fname_gensym = gensym(fname)

                    # Change the name of inner function definition to the one with gensym()
                    Turing.setfname!(fdefn_inner, fname_gensym)

                    Turing.setfname!(fdefn_inner_callback1, fname_gensym)
                    Turing.setfcall!(fdefn_inner_callback1, fname_gensym)

                    Turing.setfname!(fdefn_inner_callback2, fname_gensym)
                    Turing.setfcall!(fdefn_inner_callback2, fname_gensym)

                    Turing.setfname!(fdefn_inner_callback3, fname_gensym)
                    Turing.setfcall!(fdefn_inner_callback3, fname_gensym)

                    # insert observation values
                    @debug(fdefn_inner_callback1)
                    @debug(fdefn_inner_callback2)
                    @debug(fdefn_inner_callback3)

                    @debug fdefn_inner
               end )
	
	# check for keyword arguments
	# this should be moved into a function...
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
      		
			data_check_ex = quote
				if $_k == nothing
					@error("Data `"*$_k_str*"` is not provided.")
                else
					k_sym = Symbol($_k_str)
					if fdefn_inner.args[2].args[2].args[1].head == :line
          				insert!(fdefn_inner.args[2].args[2].args, 2, Expr(:(=), k_sym, $_k))
        			else
          				insert!(fdefn_inner.args[2].args[2].args, 1, Expr(:(=), k_sym, $_k))
        			end
				end
          	end
      		pushfirst!(fdefn_outer.args[2].args, data_check_ex)
    	end
	end

	pushfirst!(fdefn_outer.args[2].args, :(fdefn_inner = Turing._compiler_[:fdefn_inner]))

    return esc(fdefn_outer)
end

###################
# Helper function #
###################
function setfcall!(fexpr::Expr, name::Symbol)
    if fexpr.head == :function
        @assert fexpr.args[2].head == :call
        fexpr.args[2].args[1] = name
    else
        @assert length(fexpr.args) > 1
        setfcall!(fexpr.args[2], name)
    end
end

function setfname!(fexpr::Expr, name::Symbol)
    if fexpr.head == :function
        fexpr.args[1].args[1] = name
    else
        @assert length(fexpr.args) > 1
        setfname!(fexpr.args[2], name)
    end
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
