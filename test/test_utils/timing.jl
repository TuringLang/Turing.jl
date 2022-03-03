const TIMES = OrderedDict()

# To get rid of Julia's compile time.
TIMES["warmup"] = @timed 1 + 1

function time_include(path, group=nothing)
    key = isnothing(group) ? path : "$group - $path"
    TIMES[key] = @timed include(path)
end

"Return a string of pretty printed running times based on TIMES."
function running_times(times)::String
    tcopy = deepcopy(times)
    pop!(tcopy, "warmup")
    names = [first(x) for x in tcopy]
    times = [round(last(x).time; digits=1) for x in tcopy]
    allocations = [round(last(x).bytes / 10^9; digits=1) for x in tcopy]
    df = DataFrame(
        "Test" => names,
        "Allocations (GB)" => allocations,
        "Time (seconds)" => times
    )
    pretty_printed = string(df)
    return """
        $pretty_printed

        Note that the reported times differ between GitHub Runners due to different CPUs; allocations are more stable.
        """
end

function write_running_times(times)
    text = running_times(times)
    path = joinpath(pkgdir(Turing), "benchmarks", "output", "times.txt")
    mkdir(dirname(path))
    write(path, text)
    return text
end
