using Pkg

if !haskey(ENV, "CMDSTAN_HOME") || ENV["CMDSTAN_HOME"] == ""
    # Make the cmdstan home directory

    CMDSTAN = abspath(joinpath(@__DIR__, "..", "cmdstan", ))
    if !ispath(CMDSTAN)
        mkdir(CMDSTAN)
    end

    # Get cmdstan and uncompress it from its url in deps/cmdstan_url.txt

    cmdstan_url = strip(String(read("cmdstan_url.txt")))
    compressed = splitdir(cmdstan_url)[2]
    dirname = splitext(compressed)[1]
    cmdstan_home = joinpath(CMDSTAN, dirname)
    current_dir = pwd()
    cd(CMDSTAN)
    if !ispath(compressed)
        Pkg.add("HTTP")
        import HTTP
        compressedfile = HTTP.get(cmdstan_url)
        write(compressed, compressedfile.body)
    end
    Pkg.add(["ZipFile", "InfoZIP"])
    import InfoZIP
    InfoZIP.unzip(compressed, CMDSTAN)
    cd(current_dir)
    println("CMDStan is installed at path: $cmdstan_home")

    # Wrie the src file that sets ENV["CMDSTAN_HOME"]

    write(joinpath("..", "src", "cmdstan_home.jl"), "cmdstan_home() = \"$(replace(cmdstan_home, "\\"=>"\\\\"))\"")
else
    cmdstan_home_path = joinpath("..", "src", "cmdstan_home.jl")
    if ispath(cmdstan_home_path)
        rm(cmdstan_home_path)
    end
    write(joinpath("..", "src", "cmdstan_home.jl"), "cmdstan_home() = $(ENV["CMDSTAN_HOME"])")
end
