if VERSION >= v"0.5"
  if is_unix() run(`make`) end
  if is_windows() run(`mingw32-make`) end
else
  @windows_only run(`mingw32-make`) 
  @unix_only run(`make`)
end
