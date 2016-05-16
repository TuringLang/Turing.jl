try
  run(`make`)
catch x
  if isa(x, LoadError)
    run(`mingw32-make`)
  else
    throw(x)
  end
end
