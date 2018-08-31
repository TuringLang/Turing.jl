####################
# Deprecated codes #
########################################################
# TODO: remove related code of varname() in test files #
########################################################

varname(s::Symbol)  = nothing, s
varname(expr::Expr) = begin
  # Initialize an expression block to store the code for creating uid
  local sym
  @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
  indexing_ex = Expr(:block)
  # Add the initialization statement for uid
  push!(indexing_ex.args, quote indexing_list = [] end)
  # Initialize a local container for parsing and add the expr to it
  to_eval = []; pushfirst!(to_eval, expr)
  # Parse the expression and creating the code for creating uid
  find_head = false
  while length(to_eval) > 0
    evaling = popfirst!(to_eval)   # get the current expression to deal with
    if isa(evaling, Expr) && evaling.head == :ref && ~find_head
      # Add all the indexing arguments to the left
      pushfirst!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
      # Add first argument depending on its type
      # If it is an expression, it means it's a nested array calling
      # Otherwise it's the symbol for the calling
      if isa(evaling.args[1], Expr)
        pushfirst!(to_eval, evaling.args[1])
      else
        # push!(indexing_ex.args, quote pushfirst!(indexing_list, $(string(evaling.args[1]))) end)
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
