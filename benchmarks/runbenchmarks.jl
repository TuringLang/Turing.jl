using Pkg
using Dates

PROJECT_DIR = abspath(@__DIR__) |> dirname

# Check whether triggering comment contains '[benchmark]' command
#  and return immediately if not. 
# Reference implementation of listening to GH comments: 
#  https://github.com/repetitive/actions/blob/master/clojure/entrypoint.sh

# Prepare packages
Pkg.build("Turing")
Pkg.add("BenchmarkTools")

# Run `MCMCBenchmarks`
# Add code

# Report results back to PRs
# Add code
