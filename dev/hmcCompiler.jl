include("../src/samplers/support/prior.jl")

macro assume(ex)
  if isa(ex.args[2], Symbol)
    ex_assume = quote
      $(ex.args[2]) = assume(
        PriorSym(Symbol($(string(ex.args[2]))))
      )
    end
  else
    ex_assume = quote
      $(ex.args[2]) = assume(
        PriorArr(parse($(string(ex.args[2]))),Symbol($(string(ex.args[2].args[2]))),$(ex.args[2].args[2]))
      )
    end
  end
  esc(ex_assume)
end

function assume(prior :: Prior)
  # println(prior)
  return 1
end

xs = [11, 22, 33]
for i = 1:3
  @assume x ~ Normal(0, 1)
end
