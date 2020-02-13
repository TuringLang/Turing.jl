# copied from https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/doc/manual.md#defining-benchmark-suites

Random, BenchmarkTools

# Define a parent BenchmarkGroup to contain our suite
suite = BenchmarkGroup()

# Add some child groups to our benchmark suite. The most relevant BenchmarkGroup constructor
# for this case is BenchmarkGroup(tags::Vector). These tags are useful for
# filtering benchmarks by topic, which we'll cover in a later section.
suite["utf8"] = BenchmarkGroup(["string", "unicode"])
suite["trig"] = BenchmarkGroup(["math", "triangles"])

# Add some benchmarks to the "utf8" group
teststr = join(rand(MersenneTwister(1), 'a':'d', 10^4));
suite["utf8"]["replace"] = @benchmarkable replace($teststr, "a" => "b")
suite["utf8"]["join"] = @benchmarkable join($teststr, $teststr)

# Add some benchmarks to the "trig" group
for f in (sin, cos, tan)
    for x in (0.0, pi)
        suite["trig"][string(f), x] = @benchmarkable $(f)($x)
    end
end
