###################
# Helper function #
###################

function parse_indexing(expr)
  # Initialize an expression block to store the code for creating uid
  uid_ex = Expr(:block)
  # Add the initialization statement for uid
  push!(uid_ex.args, quote uid_list = [] end)
  # Initialize a local container for parsing and add the expr to it
  to_eval = []; unshift!(to_eval, expr)
  # Parse the expression and creating the code for creating uid
  while length(to_eval) > 0
    evaling = shift!(to_eval)   # get the current expression to deal with
    if isa(evaling, Expr)
      # Add all the indexing arguments to the left
      unshift!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
      # Add first argument depending on its type
      # If it is an expression, it means it's a nested array calling
      # Otherwise it's the symbol for the calling
      if isa(evaling.args[1], Expr)
        unshift!(to_eval, evaling.args[1])
      else
        push!(uid_ex.args, quote unshift!(uid_list, $(string(evaling.args[1]))) end)
      end
    else
      # Evaluting the concrete value of the indexing variable
      push!(uid_ex.args, quote push!(uid_list, string($evaling)) end)
    end
  end
  push!(uid_ex.args, quote uid = reduce(*, uid_list) end)
  uid_ex
end

function gen_assume_ex(left, right)
  # The if statement is to deterimnet how to pass the prior.
  # It only supports pure symbol and Array(/Dict) now.
  if isa(left, Symbol)  # symbol
    quote
      $(left) = Turing.assume(
        sampler,
        $(right),   # distribution
        Var(Symbol($(string(left)))),
        varInfo
      )
    end
  else                  # indexing
    uid_ex = parse_indexing(left)
    push!(
      uid_ex.args,
      quote
        $(left) = Turing.assume(
          sampler,
          $(right),   # distribution
          Var(Symbol(uid_list[1]),Symbol(uid)),
          varInfo
        )
      end
    )
    uid_ex
  end
end

insdelim(c, deli=",") = reduce((e, res) -> append!(e, [res, ","]), [], c)[1:end-1]

#################
# Overload of ~ #
#################

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
          varInfo
        )
      end
    )
  else
    _left = left  # left is the variable (symbol) itself
    # Get symbol from expr like x[1][2]
    while typeof(_left) != Symbol
      _left = _left.args[1]
    end
    left_sym = string(_left)
    # Require all data to be stored in data dictionary.
    if _left in TURING[:modelarglist]
      if ~(_left in TURING[:model_dvar_list])
        println("[Turing]: Observe - `" * left_sym * "` is an observation")
        push!(TURING[:model_dvar_list], _left)
      end
      esc(
        quote
          # Call observe
          Turing.observe(
            sampler,
            $(right),   # Distribution
            $(left),    # Data point
            varInfo
          )
        end
      )
    else
      if ~(_left in TURING[:model_pvar_list])
        msg = "[Turing]: Assume - `" * left_sym * "` is a parameter"
        isdefined(Symbol(left_sym)) && (msg  *= " (ignoring `$(left_sym)` found in global scope)")
        println(msg)
        push!(TURING[:model_pvar_list], _left)
      end

      esc(
        quote
           #if isa(Symbol($left_sym), TArray) || ~isdefined(Symbol($left_sym))
            # Call assume
            $(gen_assume_ex(left, right))
          #else
          #  throw(ErrorException("Redefining of existing variable (" * $left_sym * ") is not allowed."))
          #end
        end
      )
    end
  end
end

####################
# Modelling Syntax #
####################

doc"""
    predict(ex...)

Operation for defining the variable(s) to return.

Usage:

```julia
@predict x y z
```

Here `x`, `y`, `z` are symbols.
"""
macro predict(ex...)
  dprintln(1, "marco predicting...")
  ex_funcs = Expr(:block)
  for i = 1:length(ex)
    @assert typeof(ex[i]) == Symbol
    sym = string(ex[i])
    push!(
      ex_funcs.args,
      :(ct = current_task();
        Turing.predict(
          sampler,
          Symbol($sym), get(ct, $(ex[i]))
        )
      )
    )
  end
  esc(ex_funcs)
end

doc"""
    predictall(ex...)

Operation for return all variables depending on a `VarInfo` instance.
Usage:

```julia
@predictall vi
```

Here `vi` are is of type `VarInfo`.
"""
macro predictall(ex)
  ex_funcs = Expr(:block)
  push!(
    ex_funcs.args,
    :(ct = current_task();
      Turing.predict(
        sampler,
        get(ct, $(ex)),
        ct
      )
    )
  )
  esc(ex_funcs)
end

#################
# Main Compiler #
#################

doc"""
    @model(name, fbody)

Wrapper for models.

Usage:

```julia
@model model() begin
  body
end
```

Example:

```julia
@model gauss() begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return(s, m)
end
```
"""
macro model(name, fbody)
  dprintln(1, "marco modelling...")
  # Functions defined via model macro have an implicit varinfo array.
  # This varinfo array is useful is task cloning.

  # Turn f into f() if necessary.
  fname = isa(name, Symbol) ? Expr(:call, name) : name

  # Get parameters from the argument list
  arglist = fname.args[2:end]
  TURING[:modelarglist] = arglist
  TURING[:model_dvar_list] = Set{Symbol}() # Data
  TURING[:model_pvar_list] = Set{Symbol}() # Parameter
  fname.args = fname.args[1:1]  # remove arguments

  # Set parameters as model(data, varInfo, sampler)
  push!(fname.args, Expr(Symbol("kw"), :data, :(Dict())))
  push!(fname.args, Expr(Symbol("kw"), :varInfo, :(VarInfo())))
  push!(fname.args, Expr(Symbol("kw"), :sampler, :(Turing.sampler)))

  # Assign variables in data locally
  local_assign_ex = quote
    for k in keys(data)
      ex = Expr(Symbol("="), k, data[k])
      eval(ex)
    end
  end
  unshift!(fbody.args, local_assign_ex)

  # Generate @predict
  return_ex = fbody.args[end]   # get last statement of defined model
  if typeof(return_ex) == Symbol || return_ex.head == :return || return_ex.head == :tuple
    # Convert return(a, b, c) to @predict a b c
    predict_ex = parse("@predict " * replace(replace(string(return_ex), r"\(|\)|return", ""), ",", " "))
  else
    # If there is no variables specified, predict all
    predict_ex = parse("@predictall varInfo")
  end
  fbody.args[end] = Expr(Symbol("if"), parse("sampler != nothing"), predict_ex) # only generate predict marcos if there is a sampler exisiting, i.e. predict nothing if the model is run on its own

  # Trick for HMC sampler as it needs the return of varInfo
  push!(fbody.args, parse("if ~isa(sampler, ImportanceSampler) current_task().storage[:turing_varinfo] = varInfo end"))

  # Generate model declaration
  ex = Expr(:function, fname, fbody)
  TURING[:modelex] = ex
  return esc(ex)  # NOTE: esc() makes sure that ex is resovled where @model is called
end

doc"""
    @model(fexpr)

Wrapper for models.

Usage:

```julia
@model f() = begin
  body
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
  name = fexpr.args[1]
  fbody = fexpr.args[2].args[end] # NOTE: nested args is used here because the orignal model expr is in a block
  esc(:(@model $name $fbody))
end

doc"""
    @sample(modelcall, alg)

Macro for running the inference engine.

Usage:

```julia
@sample(modelf(params), alg)
```

Example:

```julia
@sample(gauss(x), SMC(100))
```
"""
macro sample(modelcall, alg, optionalps...)
  modelf = modelcall.args[1]      # get the function symbol for model
  psyms = modelcall.args[2:end]   # get the (passed-in) symbols
  esc(quote
    data = Dict()
    arglist = Turing.TURING[:modelarglist]
    localsyms = $psyms
    for i = 1:length(arglist)
      localsym = localsyms[i]     # passed-in symbols are local
      arg = arglist[i]            # arglist are symbols in model scope
      data[arg] = eval(localsym)
    end
    sample($modelf, data, $alg, $optionalps...)
  end)
end
