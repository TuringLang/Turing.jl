@static if is_unix() run(`make`)
@static if is_windows() run(`mingw32-make`)

