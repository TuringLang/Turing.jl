module BenchmarkHelper

export PROJECT_DIR,
    benchmark_files,
    log_report, save_results,
    @tbenchmark, @tbenchmark_expr,
    run_benchmark

PROJECT_DIR = abspath(@__DIR__) |> dirname

NON_BENCHMARK_FILES = [
    "BenchmarkHelper.jl",
    "github-action-runner.jl",
    "runbenchmarks.jl",
]

function benchmark_files()
    filter(readdir(joinpath(PROJECT_DIR, "benchmarks"))) do file
        !in(file, NON_BENCHMARK_FILES) && endswith(file, ".jl")
    end
end

function log_report(log)
    # TODO
    @show log
end

function save_results(data::Dict)
    # TODO
    @show data
end

macro tbenchmark(alg, model, data)
    model = :(($model isa String ? eval(Meta.parse($model)) : $model))
    model_dfn = (data isa Expr && data.head == :tuple) ?
        :(model_f = $model($(data)...)) : model_f = :(model_f = $model($data))
    esc(quote
        $model_dfn
        chain, t_elapsed, mem, gctime, memallocs  = @timed sample(model_f, $alg)
        Dict(
            "engine" => $(string(alg)),
            "time" => t_elapsed,
            "memory" => mem,
        )
        end)
end

macro tbenchmark_expr(engine, expr)
    quote
        chain, t_elapsed, mem, gctime, memallocs  = @timed $(esc(expr))
        Dict(
            "engine" => $(string(engine)),
            "time" => t_elapsed,
            "memory" => mem,
        )
    end
end

function run_benchmark(file)
    code = """
    using Pkg;
    pkg"instantiate; add JSON GitHub;"
    Pkg.build(verbose=true)
    push!(LOAD_PATH, joinpath("$(PROJECT_DIR)", "benchmarks"))
    using BenchmarkHelper
    include("$(file)")
    save_results(BENCHMARK_RESULT)
    """
    job = `julia --project=$(PROJECT_DIR) -e $code`
    @info(job);
    run(job)
    @info("run `$file` ✓")
end

end
