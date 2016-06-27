# Usage:
# @assume x ~ Dist
# , where x is a symbol to be used
# and Dist is a valid distribution from the Distributions package
macro assume(ex)
  @assert ex.args[1] == symbol("@~")
  esc(quote
    $(ex.args[2]) = Turing.assume(
                                  Turing.sampler,
                                  $(ex.args[3])
                                 )
  end)
end

# Usage:
# @observe(x ~ Dist)
# , where x is a value and Dist is a valid distribution
macro observe(ex)
  @assert ex.args[1] == symbol("@~")
  global TURING
  ex2 = Expr(:block, nothing)
  push!(
        ex2.args,
        :(Turing.observe(
                         Turing.sampler,
                         logpdf(
                                $(ex.args[3]),
                                $(ex.args[2])
                               )
                        )
         )
       )
  esc(ex2)
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
  return esc(ex)
end
