# Helper function
doc"""
    has_ops(ex)

Check if has optional arguments.

```julia
has_ops(parse("@assume x ~ Normal(0, 1; :static=true)"))  # gives true
has_ops(parse("@assume x ~ Normal(0, 1)"))                # gives false
has_ops(parse("@assume x ~ Binomial(; :static=true)"))    # gives true
has_ops(parse("@assume x ~ Binomial()"))                  # gives false
```
"""
function has_ops(ex)
  if length(ex.args[3].args) <= 1               # check if the D() has parameters
    return false                                # Binominal() can have empty
  elseif typeof(ex.args[3].args[2]) != Expr     # check if has optional arguments
    return false
  elseif ex.args[3].args[2].head != :parameters # check if parameters valid
    return false
  end
  true
end


# Operation for defining the prior
# Usage:
# @assume x ~ Dist
# , where x is a symbol to be used
# and Dist is a valid distribution from the Distributions package
macro assume(ex)
  dprintln(1, "marco assuming...")
  @assert ex.args[1] == symbol("@~")
  # Check if have extra arguements setting
  if has_ops(ex)
    # If static is set
    if ex.args[3].args[2].args[1].args[1] == :(:static) && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # If param is set
    if ex.args[3].args[2].args[1].args[1] == :(:param) && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # Remove the extra argument
    splice!(ex.args[3].args, 2)
  end

  sym = gensym()
  esc(
    quote
      $(ex.args[2]) = Turing.assume(
        Turing.sampler,
        $(ex.args[3]),    # Distribution
        Prior(Symbol($(string(sym))))
      )
    end
  )
end

# Operation for defining the likelihood
# Usage:
# @observe(x ~ Dist)
# , where x is a value and Dist is a valid distribution
macro observe(ex)
  dprintln(1, "marco observing...")
  @assert ex.args[1] == symbol("@~")

  global TURING
  # Check if have extra arguements setting
  if has_ops(ex)
    # If static is set
    if ex.args[3].args[2].args[1].args[1] == :(:static) && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # If param is set
    if ex.args[3].args[2].args[1].args[1] == :(:param) && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # Remove the extra argument
    splice!(ex.args[3].args, 2)
  end

  esc(
    quote
      Turing.observe(
        Turing.sampler,
        $(ex.args[3]),   # Distribution
        $(ex.args[2])    # Data point
      )
    end
  )
end

# Usage:
# @predict x y z
# , where x, y, z are symbols
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
          Turing.sampler,
          symbol($sym),
          get(ct, $(ex[i]))
        )
      )
    )
  end
  esc(ex_funcs)
end


# Usage:
# @model f body
macro model(name, fbody)
  dprintln(1, "marco modelling...")
  # Functions defined via model macro have an implicit varinfo array.
  # This varinfo array is useful is task cloning.

  # Turn f into f() if necessary.
  fname = isa(name, Symbol) ? Expr(:call, name) : name
  ex = Expr(:function, fname, fbody)

  TURING[:modelex] = ex
  return esc(ex)  # esc() makes sure that ex is resovled where @model is called
end
