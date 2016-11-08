if VERSION >= v"0.5"
  if is_unix() run(`make`) end
  if is_windows() run(`mingw32-make`) end
else
  @unix_only run(`make`)
  @windows_only run(`mingw32-make`)
end


