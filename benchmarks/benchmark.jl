using Turing

CONFIG = Dict(
  "model-list" => [
    #"naive-bayes",
    #"normal-loc",
    #"normal-mixture",
    "gauss",
    "bernoulli",
    #"negative-binomial"
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
    include("$(model).run.jl")
    println("`$model` ✓")
  end

  println("Turing benchmarking completed.")

end
