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
log2str(logd::Dict, monitor=[]) = begin
  str = ""
  str *= ("/=======================================================================") * "\n"
  str *= ("| Benchmark Result for >>> $(logd["name"]) <<<") * "\n"
  str *= ("|-----------------------------------------------------------------------") * "\n"
  str *= ("| Overview") * "\n"
  str *= ("|-----------------------------------------------------------------------") * "\n"
  str *= ("| Inference Engine  : $(logd["engine"])") * "\n"
  str *= ("| Time Used (s)     : $(logd["time"])") * "\n"
  if haskey(logd, "time_stan")
    str *= ("|   -> time by Stan : $(logd["time_stan"])") * "\n"
  end
  str *= ("| Mem Alloc (bytes) : $(logd["mem"])") * "\n"
  if haskey(logd, "turing")
    str *= ("|-----------------------------------------------------------------------") * "\n"
    str *= ("| Turing Inference Result") * "\n"
    str *= ("|-----------------------------------------------------------------------") * "\n"
    for (v, m) = logd["turing"]
      if isempty(monitor) || v in monitor
        str *= ("| >> $v <<") * "\n"
        str *= ("| mean = $(round(m, 3))") * "\n"
        if haskey(logd, "analytic") && haskey(logd["analytic"], v)
          str *= ("|   -> analytic = $(round(logd["analytic"][v], 3)), ")
          diff = abs(m - logd["analytic"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            # TODO: try to fix this
            print_with_color(:red, diff_output*"\n")
            str *= (diff_output) * "\n"
          else
            str *= (diff_output) * "\n"
          end
        end
        if haskey(logd, "stan") && haskey(logd["stan"], v)
          str *= ("|   -> Stan     = $(round(logd["stan"][v], 3)), ")
          diff = abs(m - logd["stan"][v])
          diff_output = "diff = $(round(diff, 3))"
          if sum(diff) > 0.2
            # TODO: try to fix this
            print_with_color(:red, diff_output*"\n")
            str *= (diff_output) * "\n"
          else
            str *= (diff_output) * "\n"
          end
        end
      end
    end
  end
  if haskey(logd, "note")
    str *= ("|-----------------------------------------------------------------------") * "\n"
    str *= ("| Note:") * "\n"
    note = logd["note"]
    str *= ("| $note") * "\n"
  end
  str *= ("\\=======================================================================") * "\n"
end

print_log(logd::Dict, monitor=[]) = print(log2str(logd, monitor))

send_log(logd::Dict, monitor=[]) = begin
  # log_str = log2str(logd, monitor)
  # send_str(log_str, logd["name"])
  dir_old = pwd()
  cd(Pkg.dir("Turing"))
  commit_str = replace(split(readstring(pipeline(`git show --summary `, `grep "commit"`)), " ")[2], "\n", "")
  cd(dir_old)
  time_str = "$(Dates.format(now(), "dd-u-yyyy-HH-MM-SS"))"
  logd["created"] = time_str
  logd["commit"] = commit_str
  post("https://api.mlab.com/api/1/databases/benchmark/collections/log?apiKey=Hak1H9--KFJz7aAx2rAbNNgub1KEylgN"; json=logd)
end

send_str(str::String, fname::String) = begin
  dir_old = pwd()
  cd(Pkg.dir("Turing"))
  commit_str = replace(split(readstring(pipeline(`git show --summary `, `grep "commit"`)), " ")[2], "\n", "")
  cd(dir_old)
  time_str = "$(Dates.format(now(), "dd-u-yyyy-HH-MM-SS"))"
  post("http://80.85.86.210:1110"; files = [FileParam(str, "text","upfile","benchmark-$time_str-$commit_str-$fname.txt")])
end
