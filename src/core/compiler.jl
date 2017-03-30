###################
# Helper function #
###################

insdelim(c, deli=",") = reduce((e, res) -> append!(e, [res, ","]), [], c)[1:end-1]

function varname(expr)
  # Initialize an expression block to store the code for creating uid
  local sym
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
          vi
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
            vi
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
      # The if statement is to deterimnet how to pass the prior.
      # It only supports pure symbol and Array(/Dict) now.
      if isa(left, Symbol)
        # Symbol
        assume_ex = quote
          sym = Symbol($(string(left)))
          vn = nextvn(vi, Symbol($(string(gensym()))), sym, "")
          $(left) = Turing.assume(
            sampler,
            $(right),   # dist
            vn,         # VarName
            vi          # VarInfo
          )
          ct = current_task();
          Turing.predict(sampler, sym, get(ct, $(left)))
        end
      else
        # Indexing
        assume_ex, sym = varname(left)    # sym here is used in predict()
        # NOTE:
        # The initialization of assume_ex is indexing_ex,
        # in which sym will store the variable symbol (Symbol),
        # and indexing will store the indexing (String)
        push!(
          assume_ex.args,
          quote
            vn = nextvn(vi, Symbol($(string(gensym()))), sym, indexing)
            $(left) = Turing.assume(
              sampler,
              $(right),   # dist
              vn,         # VarName
              vi          # VarInfo
            )
            ct = current_task();
            Turing.predict(sampler, sym, get(ct, $(sym)))
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

  # Set parameters as model(data, vi, sampler)
  push!(fname.args, Expr(Symbol("kw"), :data, :(Dict())))
  push!(fname.args, Expr(Symbol("kw"), :vi, :(VarInfo())))
  push!(fname.args, Expr(Symbol("kw"), :sampler, :(Turing.sampler)))

  # Assign variables in data locally
  local_assign_ex = quote
    for k in keys(data)
      ex = Expr(Symbol("="), k, data[k])
      eval(ex)
    end
  end
  unshift!(fbody.args, local_assign_ex)

  # Always return VarInfo
  return_ex = fbody.args[end]   # get last statement of defined model
  vi_ex = quote
    vi
  end
  if typeof(return_ex) == Symbol || return_ex.head == :return || return_ex.head == :tuple
    fbody.args[end] = vi_ex
  else
    push!(fbody.args, vi_ex)
  end

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
    data_ = Dict()
    arglist_ = Turing.TURING[:modelarglist]
    localsyms_ = $psyms
    for i_ = 1:length(arglist_)
      localsym_ = localsyms_[i_]     # passed-in symbols are local
      arg_ = arglist_[i_]            # arglist are symbols in model scope
      data_[arg_] = eval(localsym_)
    end
    sample($modelf, data_, $alg, $optionalps...)
  end)
end
