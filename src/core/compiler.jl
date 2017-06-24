#################
# Overload of ~ #
#################

doc"""
    var_name ~ Distribution()

`~` notation is to specifiy *a variable follows a distributions*.

If `var_name` is an un-defined variable or a container (e.g. Vector or Matrix), this variable will be treated as model parameter; otherwise if `var_name` is defined, this variable will be treated as data.
"""
macro ~(left, right)
  # Is multivariate a subtype of real, e.g. Vector, Matrix?
  if isa(left, Real)                  # value
    # Call observe
    esc(
      quote
        Turing.observe(
          sampler,
          $(right),   # Distribution
          $(left),    # Data point
          vi
        )
      end
    )
  else
    _, vsym = varname(left)
    vsym_str = string(vsym)
    # Require all data to be stored in data dictionary.
    if vsym in Turing._compiler_[:fargs]
      if ~(vsym in Turing._compiler_[:dvars])
        dprintln(FCOMPILER, " Observe - `" * vsym_str * "` is an observation")
        push!(Turing._compiler_[:dvars], vsym)
      end
      esc(
        quote
          # Call observe
          Turing.observe(
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
        isdefined(Symbol(vsym_str)) && (msg  *= " (ignoring `$(vsym_str)` found in global scope)")
        dprintln(FCOMPILER, msg)
        push!(Turing._compiler_[:pvars], vsym)
      end
      # The if statement is to deterimnet how to pass the prior.
      # It only supports pure symbol and Array(/Dict) now.
      #csym_str = string(gensym())
      if isa(left, Symbol)
        # Symbol
        assume_ex = quote
          csym_str = string(Turing._compiler_[:fname])*"_var"* string(@__LINE__)
          sym = Symbol($(string(left)))
          vn = Turing.VarName(vi, Symbol(csym_str), sym, "")
          if isa($(right), Vector)
            $(left) = Turing.assume(
              sampler,
              $(right),   # dist
              vn,         # VarName
              $(left),
              vi          # VarInfo
            )
          else
            $(left) = Turing.assume(
              sampler,
              $(right),   # dist
              vn,         # VarName
              vi          # VarInfo
            )
          end
        end
      else
        # Indexing
        assume_ex, sym = varname(left)    # sym here is used in predict()
        # NOTE:
        # The initialization of assume_ex is indexing_ex,
        # in which sym will store the variable symbol (Symbol),
        # and indexing will store the indexing (String)
        # csym_str = string(gensym())
        push!(
          assume_ex.args,
          quote
            csym_str = string(Turing._compiler_[:fname]) * string(@__LINE__)
            vn = Turing.VarName(vi, Symbol(csym_str), sym, indexing)
            $(left) = Turing.assume(
              sampler,
              $(right),   # dist
              vn,         # VarName
              vi          # VarInfo
            )
          end
        )
      end
      esc(assume_ex)
    end
  end
end

#################
# Main Compiler #
#################

doc"""
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
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return(s, m)
end
```
"""
macro model(fexpr)
   # Compiler design: sample(fname_compiletime(x,y), sampler)
   #   fname_compiletime(x=nothing,y=nothing; data=data,compiler=compiler) = begin
   #      ex = quote
   #          fname_runtime(;vi=VarInfo,sampler=nothing) = begin
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


  dprintln(1, fexpr)

  fname = fexpr.args[1].args[1]      # Get model name f
  fargs = fexpr.args[1].args[2:end]  # Get model parameters (x,y;z=..)
  fbody = fexpr.args[2].args[end]    # NOTE: nested args is used here because the orignal model expr is in a block

  # Prepare for keyword arguments, e.g.
  #   f(x,y)
  #       ==> f(x,y;)
  #   f(x,y; c=1)
  #       ==> unchanged
  if (length(fargs) == 0 ||         # e.g. f()
          isa(fargs[1], Symbol) ||  # e.g. f(x,y)
          fargs[1].head == :kw)     # e.g. f(x,y=1)
    insert!(fargs, 1, Expr(:parameters))
  end

  dprintln(1, fname)
  dprintln(1, fargs)
  dprintln(1, fbody)

  # Remove positional arguments from inner function, e.g.
  #  f((x,y; c=1)
  #      ==> f(; c=1)
  #  f(x,y;)
  #      ==> f(;)
  fargs_inner = deepcopy(fargs)[1:1]

  # Add keyword arguments, e.g.
  #  f(; c=1)
  #      ==> f(; c=1, :vi=VarInfo(), :sample=nothing)
  #  f(;)
  #      ==> f(; :vi=VarInfo(), :sample=nothing)
  push!(fargs_inner[1].args, Expr(:kw, :vi, :(Turing.VarInfo())))
  push!(fargs_inner[1].args, Expr(:kw, :sampler, :(nothing)))
  dprintln(1, fargs_inner)

  # Modify fbody, so that we always return VarInfo
  fbody_inner = deepcopy(fbody)
  return_ex = fbody.args[end]   # get last statement of defined model
  if typeof(return_ex) == Symbol ||
       return_ex.head == :return ||
       return_ex.head == :tuple
    pop!(fbody_inner.args)
  end
  push!(fbody_inner.args, Expr(:return, :vi))
  dprintln(1, fbody_inner)

  suffix = gensym()
  fname_inner = Symbol("$(fname)_model_$suffix")
  fdefn_inner = Expr(:function, Expr(:call, fname_inner)) # fdefn = :( $fname() )
  push!(fdefn_inner.args[1].args, fargs_inner...)   # set parameters (x,y;data..)
  push!(fdefn_inner.args, deepcopy(fbody_inner))    # set function definition
  dprintln(1, fdefn_inner)

  compiler = Dict(:fname => fname,
                  :fargs => fargs,
                  :fbody => fbody,
                  :dvars => Set{Symbol}(),  # data
                  :pvars => Set{Symbol}(),  # parameter
                  :fdefn_inner => fdefn_inner)

  # Outer function defintion 1: f(x,y) ==> f(x,y;data=Dict())
  fargs_outer = deepcopy(fargs)
  # Add data argument to outer function
  push!(fargs_outer[1].args, Expr(:kw, :data, :(Dict{Symbol,Any}())))
  # Add data argument to outer function
  push!(fargs_outer[1].args, Expr(:kw, :compiler, compiler))
  for i = 2:length(fargs_outer)
    s = fargs_outer[i]
    if isa(s, Symbol)
      fargs_outer[i] = Expr(:kw, s, :nothing) # turn f(x;..) into f(x=nothing;..)
    end
  end

  fdefn_outer = Expr(:function, Expr(:call, fname, fargs_outer...),
                        Expr(:block, Expr(:return, fname_inner)))

  unshift!(fdefn_outer.args[2].args, :(Main.eval(fdefn_inner)))
  unshift!(fdefn_outer.args[2].args,  quote
      # Check fargs, data
      eval(Turing, :(_compiler_ = deepcopy($compiler)))
      fargs    = Turing._compiler_[:fargs];
      fdefn_inner   = Turing._compiler_[:fdefn_inner];
      # Copy data dictionary
      for k in keys(data)
        if fdefn_inner.args[2].args[1].head == :line
          # Preserve comments, useful for debuggers to
          # correctly locate source code oringin.
          insert!(fdefn_inner.args[2].args, 2, Expr(:(=), Symbol(k), data[k]))
        else
          insert!(fdefn_inner.args[2].args, 1, Expr(:(=), Symbol(k), data[k]))
        end
      end
      dprintln(1, fdefn_inner)
  end )

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
      dprintln(1, _k_str, " = ", _k)
      _ = quote
            if haskey(data, keytype(data)($_k_str))
              if nothing != $_k
                Turing.dwarn(0, " parameter "*$_k_str*" found twice, value in data dictionary will be used.")
              end
            else
              data[keytype(data)($_k_str)] = $_k
              data[keytype(data)($_k_str)] == nothing && Turing.derror(0, "Data `"*$_k_str*"` is not provided.")
            end
          end
      unshift!(fdefn_outer.args[2].args, _)
    end
  end
  unshift!(fdefn_outer.args[2].args, quote data = copy(data) end)

  dprintln(1, esc(fdefn_outer))
  esc(fdefn_outer)
end



###################
# Helper function #
###################

insdelim(c, deli=",") = reduce((e, res) -> append!(e, [res, ","]), [], c)[1:end-1]

varname(s::Symbol)  = nothing, s
varname(expr::Expr) = begin
  # Initialize an expression block to store the code for creating uid
  local sym
  @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
  indexing_ex = Expr(:block)
  # Add the initialization statement for uid
  push!(indexing_ex.args, quote indexing_list = [] end)
  # Initialize a local container for parsing and add the expr to it
  to_eval = []; unshift!(to_eval, expr)
  # Parse the expression and creating the code for creating uid
  find_head = false
  while length(to_eval) > 0
    evaling = shift!(to_eval)   # get the current expression to deal with
    if isa(evaling, Expr) && evaling.head == :ref && ~find_head
      # Add all the indexing arguments to the left
      unshift!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
      # Add first argument depending on its type
      # If it is an expression, it means it's a nested array calling
      # Otherwise it's the symbol for the calling
      if isa(evaling.args[1], Expr)
        unshift!(to_eval, evaling.args[1])
      else
        # push!(indexing_ex.args, quote unshift!(indexing_list, $(string(evaling.args[1]))) end)
        push!(indexing_ex.args, quote sym = Symbol($(string(evaling.args[1]))) end) # store symbol in runtime
        find_head = true
        sym = evaling.args[1] # store symbol in compilation time
      end
    else
      # Evaluting the concrete value of the indexing variable
      push!(indexing_ex.args, quote push!(indexing_list, string($evaling)) end)
    end
  end
  push!(indexing_ex.args, quote indexing = reduce(*, indexing_list) end)
  indexing_ex, sym
end
