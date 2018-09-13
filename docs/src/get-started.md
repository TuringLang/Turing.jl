# Getting Started

## Installation

To use Turing, you need install Julia first and then install Turing.

### Install Julia

You will need Julia 1.0, which you can get from [the official Julia website](http://julialang.org/downloads/).

There are three options for users:

1. A command line version [Julia/downloads](http://julialang.org/downloads/) (**recommended**).
2. A community maintained IDE [Juno](http://www.junolab.org/).
3. [JuliaBox.com](https://www.juliabox.com/) – a Jupyter notebook in the browser.

For the command line version, we recommend that you install a version downloaded from Julia's [official website](http://julialang.org/downloads/), as Turing may not work correctly with Julia provided by other sources (e.g. Turing does not work with Julia installed via apt-get due to missing header files).

Juno will also the command line version installed. This IDE is recommended for heavy users who require features like debugging, quick documentation check, etc.

JuliaBox provides a pre-installed Jupyter notebook for Julia. You can take a shot at Turing without installing Julia on your machine in few seconds.

### Install Turing.jl

Turing is an officially registered Julia package, so the following will install a stable version of Turing while inside Julia's package manager (press `]` from the REPL):

```julia
add Turing
```

If you want to use the latest version of Turing with some experimental features, you can try the following instead:

```julia
add Turing#master
test Turing
```

If all tests pass, you're ready to start using Turing.
