macro change_operation(ex)
  # Change the operation to mutiplication
  ex.args[1] = :*
  # Return an expression to print the result
  return :(println($(ex)))
end

ex = macroexpand(:(@change_operation 1 + 2 ))
