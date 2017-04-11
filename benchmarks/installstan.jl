run(`wget https://github.com/stan-dev/cmdstan/releases/download/v2.14.0/cmdstan-2.14.0.zip`)
run(`unzip -qq cmdstan-2.14.0.zip -d cmdstan`)
CMDSTAN_HOME = pwd() * "/cmdstan"
Pkg.add("Homebrew")
Pkg.add("DualNumbers")
Pkg.add("Stan")
Pkg.build("Stan")
