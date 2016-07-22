macro assume(ex)
  @assert ex.args[1] == symbol("@~")
  # Check if have extra arguements setting
  if typeof(ex.args[3].args[2]) == Expr &&  ex.args[3].args[2].head == :parameters
    # If static is set
    if ex.args[3].args[2].args[1].args[1] == :static && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # If param is set
    if ex.args[3].args[2].args[1].args[1] == :param && ex.args[3].args[2].args[1].args[2] == :true
      # Do something
    end
    # Remove the extra argument
    splice!(ex.args[3].args, 2)
  end
  # Turn Distribution type to dDistribution
  ex.args[3].args[1] = symbol("d$(ex.args[3].args[1])")

  # TODO: Finish the evaluation of index
  # Evaluate reference if necessay
  # if !isa(ex.args[2], Symbol) && ex.args[2].head == :ref
  #   ex.args[2].args[2] = eval(ex.args[2].args[2])
  # end

  # Change variable to symbol
  sym = string(ex.args[2])
  println(sym)
  # esc(quote
  #   $(ex.args[2]) = Turing.assume(
  #                                 Turing.sampler,
  #                                 $(ex.args[3]),    # dDistribution
  #                                 symbol($(sym))    # Symbol of prior
  #                                )
  #     end)
end

xs = [4, 5, 6]
for i = 1:3
  println(i)
  @assume x[i] ~ Normal(0, 1)
end

# NOTE: The problem is that the marco is converted at the compling time but the loop is only executed at the running time. Question = Can I fetch values from the running time in a marco?
