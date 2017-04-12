using Distributions
using Turing
using Stan

include("ASCIIPlot.jl");

# Run benchmark
tbenchmark(alg::String, model::String, data::String) = begin
  chain, time, mem, _, _  = eval(parse("@timed sample($model($data), $alg)"))
  alg, time, mem, chain
end

# Build logd from Turing chain
build_logd(name::String, engine::String, time, mem, tchain) = begin
  Dict(
    "name" => name,
    "engine" => engine,
    "time" => time,
    "mem" => mem,
    "turing" => Dict(v => mean(tchain[Symbol(v)]) for v in keys(tchain))
  )
end

# Log function
print_log(logd::Dict) = begin
  println("/=======================================================================")
  println("| Benchmark Result for >>> $(logd["name"]) <<<")
  println("|-----------------------------------------------------------------------")
  println("| Overview")
  println("|-----------------------------------------------------------------------")
  println("| Inference Engine  : $(logd["engine"])")
  println("| Time Used (s)     : $(logd["time"])")
  println("| Mem Alloc (bytes) : $(logd["mem"])")
  if haskey(logd, "turing")
    println("|-----------------------------------------------------------------------")
    println("| Turing Inference Result")
    println("|-----------------------------------------------------------------------")
    for (v, m) = logd["turing"]
      println("|")
      println("| E[$v] = $m")
      println("|")
      if haskey(logd, "analytic") && haskey(v ,logd["analytic"])
        println("| -> analytic = $(logd["analytic"][v])")
        println("|    diff     = $(abs(m - logd["analytic"][v]))")
      end
      if haskey(logd, "stan") && haskey(v ,logd["stan"])
        println("| -> Stan = $(logd["stan"][v])")
        println("|    diff = $(abs(m - logd["stan"][v]))")
      end
    end
  end
  println("\\=======================================================================")
end

# NOTE: put Stan models before Turing ones if you want to compare them in print_log
CONFIG = Dict(
  "model-list" => [
    #"naive-bayes",
    #"normal-loc",
    # "simple-normal-mixture-stan",
    # "simple-normal-mixture",
    "simplegauss-stan",
    "simplegauss",
    # "gauss",
    "bernoulli-stan",
    "bernoulli",
    # "gdemo-geweke"
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
