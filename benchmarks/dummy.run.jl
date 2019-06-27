using Turing, TuringBenchmarks

data = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

@model constrained_test(obs) = begin
    p ~ Beta(2,2)
    for i = 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    p
end

bench_res = @tbenchmark(HMC(1000, 1.5, 3), constrained_test, data)

# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("Dummy-Benchmark", bench_res...)

print_log(logd)
send_log(logd)
