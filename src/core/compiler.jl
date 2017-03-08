###################
# Helper function #
###################

function gen_assume_ex(left, right)
  # The if statement is to deterimnet how to pass the prior.
  # It only supports pure symbol and Array(/Dict) now.
  if isa(left, Symbol)
    quote
      $(left) = Turing.assume(
        sampler,
        $(right),   # distribution
        Var(        # pure Symbol
          Symbol($(string(left))),
          Symbol($(string(left)))
        ),
        varInfo
      )
    end
  elseif length(left.args) == 2 && isa(left.args[1], Symbol)
    quote
      $(left) = Turing.assume(
        sampler,
        $(right),   # distribution
        Var(        # array assignment
          Symbol($(string(left.args[1]))),  # pure symbol, e.g. :p for p[1]
          parse($(string(left))),           # indexing expr
          Symbol($(string(left.args[2]))),  # index symbol
          $(left.args[2])                   # index value
        ),
        varInfo
      )
    end
  elseif length(left.args) == 2 && isa(left.args[1], Expr)
    quote
      $(left) = Turing.assume(
        sampler,
        $(right),   # dDistribution
        Var(        # array assignment
          Symbol($(string(left.args[1].args[1]))),  # pure symbol
          parse($(string(left))),           # indexing expr
          Symbol($(string(left.args[1].args[2]))),  # index symbol
          $(left.args[1].args[2]),                  # index value
          Symbol($(string(left.args[2]))),  # index symbol
          $(left.args[2])                   # index value
        ),
        varInfo
      )
    end
  elseif length(left.args) == 3
    quote
      $(left) = Turing.assume(
        sampler,
        $(right),   # dDistribution
        Var(        # array assignment
          Symbol($(string(left.args[1]))),  # pure symbol
          parse($(string(left))),           # indexing expr
          Symbol($(string(left.args[2]))),  # index symbol
          $(left.args[2]),                  # index value
          Symbol($(string(left.args[3]))),  # index symbol
          $(left.args[3])                   # index value
        ),
        varInfo
      )
    end
  end
end

macro isdefined(variable)
  esc(quote
    try
      $variable
      true
    catch
      false
    end
  end)
end

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
    esc(
      quote
        # Require all data to be stored in data dictionary.
        if haskey(data, Symbol($left_sym))
          # $(_left) = data[Symbol($left_sym)]
          # Call observe
          Turing.observe(
            sampler,
            $(right),   # Distribution
            $(left),    # Data point
            varInfo
          )
        elseif @isdefined($left)
          throw(ErrorException("Redefiining of existing variable (local or global) (" * $left_sym * ") is not allowed."))
        elseif ~isdefined(Symbol($left_sym))
          # Call assume
          $(gen_assume_ex(left, right))
        else
          throw(ErrorException("Unexpted error (compiler, probably caused by @isdefined)."))
        end
      end
    )
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
    @sample(fexpr)

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
macro sample(modelcall, alg)
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
    sample($modelf, data, $alg)
  end)
end
