try
  run(`make`)
catch x
  # todo: the error should be checked here, but I don't know how to do it
  run(`mingw32-make`)
end
