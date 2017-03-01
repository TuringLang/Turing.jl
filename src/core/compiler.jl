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
        $(right),    # dDistribution
        Var(          # Pure Symbol
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
        $(right),    # dDistribution
        Var(          # Array assignment
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
        $(right),    # dDistribution
        Var(          # Array assignment
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
        $(right),    # dDistribution
        Var(          # Array assignment
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
 quote
   try
    $(esc(variable))
    true
   catch
    false
   end
 end
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
    _left = left
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
        else
          # Call assume
          $(gen_assume_ex(left, right))
        end
      end
    )
  end
end

######################
# Modelling Language #
######################

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

doc"""
    model(name, fbody)

Wrapper for models.

Usage:

```julia
@model f body
```

Example:

```julia
@model gauss begin
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end
```
"""
macro model(name, fbody)
  # name = model_ex.args[1]
  # fbody = model_ex.args[2]
  dprintln(1, "marco modelling...")
  # Functions defined via model macro have an implicit varinfo array.
  # This varinfo array is useful is task cloning.

  # Turn f into f() if necessary.
  fname = isa(name, Symbol) ? Expr(:call, name) : name
  # TODO: get parameters from the argument list
  arglist = fname.args[2:end]
  # TODO: remove arguments
  fname.args = fname.args[1:1]

  if length(find(arg -> isa(arg, Expr) && arg.head == :kw && arg.args[1] == :data, fname.args)) == 0
    push!(fname.args, Expr(Symbol("kw"), :data, :(Dict())))
  end

  push!(fname.args, Expr(Symbol("kw"), :varInfo, :(VarInfo())))
  push!(fname.args, Expr(Symbol("kw"), :sampler, :(Turing.sampler)))

  local_assign_ex = quote
    for k in keys(data)
      ex = Expr(Symbol("="), k, data[k])
      eval(ex)
    end
  end
  unshift!(fbody.args, local_assign_ex)

  # predict_ex = quote
  #   ct = current_task()
  #   ct.storage[:turing_predicts] = Dict{Symbol,Any}()
  #   for sym in syms(varInfo)
  #     ct.storage[:turing_predicts][Symbol(string(sym))] = get(ct, sym)
  #   end
  # end
  # push!(fbody.args, predict_ex)

  # return varInfo if sampler is nothing otherwise varInfo
  return_ex = fbody.args[end]   # get last statement of model
  if typeof(return_ex) == Symbol || return_ex.head == :return || return_ex.head == :tuple
    predict_ex = parse("@predict " * replace(replace(string(return_ex), r"\(|\)|return", ""), ",", " "))
  else
    predict_ex = parse("@predictall varInfo")
  end
  fbody.args[end] = Expr(Symbol("if"), parse("sampler != nothing"), predict_ex)
  push!(fbody.args, parse("if ~isa(sampler, ImportanceSampler) current_task().storage[:turing_varinfo] = varInfo end"))
  # push!(fbody.args, Expr(Symbol("if"), parse("sampler == nothing"), return_ex, :(varInfo)))

  ex = Expr(:function, fname, fbody)
  TURING[:modelex] = ex
  return esc(ex)  # esc() makes sure that ex is resovled where @model is called
end

macro sample(modelcall, alg)
  # println(typeof(modelcall))
  modelf = modelcall.args[1]
  modelt = modelf
  psyms = modelcall.args[2:end]
  # println(psyms)

  # res = sample(modelt, data, eval(alg))
  # print(res)
  # esc(:(sample($modelt, data, $alg)))
  esc(quote
    data = Dict()
    for sym in $psyms
      data[sym] = eval(sym)
    end
    sample($modelt, data, $alg)
  end)
end
