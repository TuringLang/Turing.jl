using Pkg;

if "Libtask" in keys(Pkg.installed())
    @info("Libtask is already installed.")
else
    pkg"add https://github.com/TuringLang/Libtask.jl"
end
