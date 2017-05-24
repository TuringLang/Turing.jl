# Get running time of Stan
get_stan_time(stan_model_name::String) = begin
  s = readlines(pwd()*"/tmp/$(stan_model_name)_samples_1.csv")
  m = match(r"(?<time>[0-9].[0-9]*)", s[end-1])
  float(m[:time])
end

# Run benchmark
tbenchmark(alg::String, model::String, data::String) = begin
  chain, time, mem, _, _  = eval(parse("@timed sample($model($data), $alg)"))
  alg, sum(chain[:elapsed]), mem, chain, deepcopy(chain)
end

# Build logd from Turing chain
build_logd(name::String, engine::String, time, mem, tchain, _) = begin
  Dict(
    "name" => name,
    "engine" => engine,
    "time" => time,
    "mem" => mem,
    "turing" => Dict(v => mean(tchain[Symbol(v)]) for v in keys(tchain))
  )
end

# Log function
print_log(logd::Dict, monitor=[]) = begin
  println("/=======================================================================")
  println("| Benchmark Result for >>> $(logd["name"]) <<<")
  println("|-----------------------------------------------------------------------")
  println("| Overview")
  println("|-----------------------------------------------------------------------")
  println("| Inference Engine  : $(logd["engine"])")
  println("| Time Used (s)     : $(logd["time"])")
  if haskey(logd, "time_stan")
    println("|   -> time by Stan : $(logd["time_stan"])")
  end
  println("| Mem Alloc (bytes) : $(logd["mem"])")
  if haskey(logd, "turing")
    println("|-----------------------------------------------------------------------")
    println("| Turing Inference Result")
    println("|-----------------------------------------------------------------------")
    for (v, m) = logd["turing"]
      if isempty(monitor) || v in monitor
        println("| >> $v <<")
        println("| mean = $(round(m, 3))")
        if haskey(logd, "analytic") && haskey(logd["analytic"], v)
          print("|   -> analytic = $(round(logd["analytic"][v], 3)), ")
          diff = abs(m - logd["analytic"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            print_with_color(:red, diff_output*"\n")
          else
            println(diff_output)
          end
        end
        if haskey(logd, "stan") && haskey(logd["stan"], v)
          print("|   -> Stan     = $(round(logd["stan"][v], 3)), ")
          diff = abs(m - logd["stan"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            print_with_color(:red, diff_output*"\n")
          else
            println(diff_output)
          end
        end
      end
    end
  end
  println("\\=======================================================================")
end
