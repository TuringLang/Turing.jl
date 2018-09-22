using Pkg

if !haskey(ENV, "CMDSTAN_HOME") || ENV["CMDSTAN_HOME"] == ""
    # Make the cmdstan home directory

    CMDSTAN = joinpath(@__DIR__, "..", "cmdstan", )
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

    write(joinpath("..", "src", "cmdstan_home.jl"), "ENV[\"CMDSTAN_HOME\"] = $cmdstan_home")
end
