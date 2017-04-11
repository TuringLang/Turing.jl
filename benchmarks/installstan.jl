run(`wget https://github.com/stan-dev/cmdstan/releases/download/v2.14.0/cmdstan-2.14.0.zip`)
run(`unzip -qq cmdstan-2.14.0.zip`)
CMDSTAN_HOME = pwd() * "/cmdstan-2.14.0"
println("CMDStan is installed at path: $CMDSTAN_HOME")
@osx Pkg.add("Homebrew")
Pkg.add("Stan")
Pkg.build("Stan")
