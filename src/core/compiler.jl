# Usage:
# @assume x ~ Dist
# , where x is a symbol to be used
# and Dist is a valid distribution from the Distributions package
macro assume(ex)
  @assert ex.args[1] == symbol("@~")
  if typeof(ex.args[3].args[2]) == Expr   # Check if have extra arguements
    if ex.args[3].args[2].args[1].args[1] == :static && ex.args[3].args[2].args[1].args[2] == :true     # If static is set,
      # do something
    end
    if ex.args[3].args[2].args[1].args[1] == :param && ex.args[3].args[2].args[1].args[2] == :true    # If param is set,
      # do something
    end
    splice!(ex.args[3].args, 2)   # remove this argument
    sym = string(ex.args[2])
    # TODO Replace all Distribution type using my own wrapper.
    # ex.args[3].args[1] = symbol("hmc$(ex.args[3].args[1])")
    esc(quote
      $(ex.args[2]) = Turing.assume(
                                    Turing.sampler,
                                    $(ex.args[3]),
                                    symbol($(sym))   # the symbol of prior
                                   )
    end)
  else
    esc(quote
      $(ex.args[2]) = Turing.assume(
                                    Turing.sampler,
                                    $(ex.args[3])
                                   )
    end)

  end

end

# Usage:
# @observe(x ~ Dist)
# , where x is a value and Dist is a valid distribution
macro observe(ex)
  @assert ex.args[1] == symbol("@~")
  global TURING
  ex2 = Expr(:block, nothing)

  if typeof(ex.args[3].args[2]) == Expr   # Check if have extra arguements
    if ex.args[3].args[2].args[1].args[1] == :static && ex.args[3].args[2].args[1].args[2] == :true     # If static is set,
      # Map Distribution to hmcDistribution
      ex.args[3].args[1] = symbol("hmc$(ex.args[3].args[1])")
    end
    splice!(ex.args[3].args, 2)   # remove this argument
    push!(
          ex2.args,
          :(Turing.observe(
                           Turing.sampler,
                           log(
                               hmcpdf(
                                      $(ex.args[3]),          # distribution
                                      Dual($(ex.args[2]), 0)  # data point
                                     )
                              )
                          )
           )
         )
    esc(ex2)
  else
    # Original version
    push!(
          ex2.args,
          :(Turing.observe(
                           Turing.sampler,
                           logpdf(
                                  $(ex.args[3]),  # distribution
                                  $(ex.args[2])   # data point
                                 )
                          )
           )
         )
  end
end

# Usage:
# @predict x y z
# , where x, y, z are symbols
macro predict(ex...)
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
  # println(ex_funcs)
  esc(ex_funcs)
end


# Usage:
# @model f body
macro model(name, fbody)
  # Functions defined via model macro have an implicit varinfo array.
  # This varinfo array is useful is task cloning.

  fname = isa(name, Symbol) ? Expr(:call, name) : name # Turn f into f() if necessary.
  ex = Expr(:function, fname, fbody)

  push!(
        ex.args[2].args,
        :(produce( Val{:done}) )
       )

  TURING[:modelex] = ex
  return esc(ex)  # esc() makes sure that ex is resovled where @model is called
end
