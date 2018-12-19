using Test

# Helper function for numerical tests
function check_numerical(
  chain,
  symbols::Vector{Symbol},
  exact_vals::Vector;
  eps=0.2,
)
	for (sym, val) in zip(symbols, exact_vals)
		E = mean(chain[sym])
		print("  $sym = $E ≈ $val (eps = $eps) ?")
		cmp = abs.(sum(mean(chain[sym]) - val)) <= eps
		if cmp
			printstyled("./\n", color = :green)
			printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :green)
		else
		printstyled(" X\n", color = :red)
		printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :red)
	end
  end
end

"""
  getteststoskip(filepath)

Returns a list of files (tests) to skip within a test group.
"""
function getteststoskip(filepath)
  if isfile(filepath)
	lines = readlines(filepath)
	lines = filter(line -> endswith(line, ".jl"), lines)
	lines = filter(line -> !startswith(line, "#"), lines)
	lines = filter(line -> length(split(line)) == 1, lines)
	lines = map(line -> strip(line), lines)
	return Set{String}(lines)
  else
	return Set{String}()
  end
end

function insdelim(c, deli=",")
	return reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end

"""
  runtests(; test_folders = ["all"], exclude = [])

Run specified Turing tests. By default this function runs all Turing tests.
The user can specify a list of test sets that she wants to run if necessary.
Specific files can be excluded by adding paths to the `exclude` argument.

Example:
```julia
runtests(test_folders = ["util.jl", "trace.jl"])
```

Excluding a specific file:
```julia
runtests(exclude = [joinpath("hmc.jl", "matrix_support.jl"])
```

Running a specific test:
```julia
runtests(specific_tests = [joinpath("hmc.jl", "matrix_support.jl"])
```

Note that calling `runtests` with a nonzero length `specific_tests` argument
overwrites all other function arguments.
"""
function runtests(; test_folders = ["all"], exclude = [], specific_tests = [])
	# test groups
	CORE_TESTS = ["ad.jl", "compiler.jl", "container.jl", "varinfo.jl",
		# "io.jl",
		"util.jl"]
	SAMPLER_TESTS = ["resample.jl", "adapt.jl", "vectorisation.jl", "gibbs.jl", "nuts.jl",
		"hmcda.jl", "hmc_core.jl", "hmc.jl", "sghmc.jl", "sgld.jl", "is.jl",
		"mh.jl",
		# "pmmh.jl", "ipmcmc.jl", "pgibbs.jl", "smc.jl"
		]
	TRACE_TESTS = ["trace.jl"]
	ALL = union(CORE_TESTS, SAMPLER_TESTS, TRACE_TESTS)

	# test groups that should be executed
	TEST_GROUPS = "all" ∈ test_folders ? ALL : test_folders

	# If we have specific tests, only run those.
	has_specific_tests = length(specific_tests) > 0
	if has_specific_tests
		TEST_GROUPS = []
		for i in specific_tests
			TEST_GROUPS = TEST_GROUPS ∪ [dirname(i)]
		end
	end

	# Run tests
	tr = @testset "Turing tests" begin
	for test_group in TEST_GROUPS
		teststoskip = getteststoskip(joinpath(test_group, "skip_tests"))
			@testset "$(test_group)" begin
				for test in filter(f -> endswith(f, ".jl")
					&& !(f ∈ teststoskip), readdir(test_group))

					@testset "$(test)" begin
						test_path = joinpath(test_group, test)

						# If we've got specific tests, examine those.
						if has_specific_tests
							if test_path in specific_tests
								include(joinpath(test_group, test))
							end
						# Otherwise look at every file in the folder.
						else
							if test_path in exclude
								@info "Skipping $test_path."
							else
								include(joinpath(test_group, test))
							end
						end

					end
				end
			end
		end
	end

	println()
	Test.print_test_results(tr)
end

using Pkg;
"""
	isinstalled(x::String)
Check if a package is installed.
"""
isinstalled(x::AbstractString) = x ∈ keys(Pkg.installed())


# NOTE: Remove the code below when DynamicHMC is registered.
using Pkg;
isinstalled("TransformVariables") || pkg"add https://github.com/tpapp/TransformVariables.jl#master";
isinstalled("LogDensityProblems") || pkg"add https://github.com/tpapp/LogDensityProblems.jl#master";
isinstalled("DynamicHMC") || pkg"add https://github.com/tpapp/DynamicHMC.jl#master";
