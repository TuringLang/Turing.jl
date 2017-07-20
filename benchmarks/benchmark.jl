using Distributions
using Turing
using Stan

# NOTE: put Stan models before Turing ones if you want to compare them in print_log
CONFIG = Dict(
  "model-list" => [
    "gdemo-geweke",
    #"normal-loc",
    "normal-mixture",
    "gdemo",
    "gauss",
    "bernoulli",
    #"negative-binomial",
    "school8",
    "binormal",
    "kid"
  ],

  "test-level" => 2   # 1 = model lang, 2 = whole interface
)

if CONFIG["test-level"] == 1

  println("Turing compiler test started.")

  for model in CONFIG["model-list"]
    println("Tesing `$model` ... ")
    include("$(model).run.jl")
    println("✓")
  end

  println("Turing compiler test passed.")

elseif CONFIG["test-level"] == 2

  println("Turing benchmarking started.")

  for model in CONFIG["model-list"]
    println("Benchmarking `$model` ... ")
    job = `julia -e " cd(\"$(pwd())\");include(dirname(\"$(@__FILE__)\")*\"/benchmarkhelper.jl\");
                         CMDSTAN_HOME = \"$CMDSTAN_HOME\";
                         using Turing, Distributions, Stan;
                         include(dirname(\"$(@__FILE__)\")*\"/$(model).run.jl\") "`
    println(job); run(job)
    println("`$model` ✓")
  end

  println("Turing benchmarking completed.")

end
