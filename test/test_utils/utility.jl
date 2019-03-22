using Test

# Helper function for numerical tests
function check_numerical(
  chain,
  symbols::Vector,
  exact_vals::Vector;
  eps=0.2,
)
	for (sym, val) in zip(symbols, exact_vals)
        @info sym, val
        E = val isa Real ? mean(chain[sym].value) : vec(mean(chain[sym].value, dims=[1]))
		print("  $sym = $E ≈ $val (eps = $eps) ?")
		cmp = abs.(sum(E - val)) <= eps
		if cmp
			printstyled("./\n", color = :green)
			printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :green)
		else
		printstyled(" X\n", color = :red)
		printstyled("    $sym = $E, diff = $(abs.(E - val))\n", color = :red)
	end
  end
end

# Wrapper function to quickly check gdemo accuracy.
function check_gdemo(chain; eps = 0.2)
    check_numerical(chain, [:s, :m], [49/24, 7/6], eps=eps)
end

function insdelim(c, deli=",")
	return reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end

# Include each file in a directory.
function include_dir(path::AbstractString)
    for (root, dirs, files) in walkdir(path)
        for file in files
            include(joinpath(root, file))
        end
    end
end

using Pkg;
"""
	isinstalled(x::String)
Check if a package is installed.
"""
isinstalled(x::AbstractString) = x ∈ keys(Pkg.installed())


# # NOTE: Remove the code below when DynamicHMC is registered.
# using Pkg;
# isinstalled("TransformVariables") || pkg"add https://github.com/tpapp/TransformVariables.jl#master";
# isinstalled("LogDensityProblems") || pkg"add https://github.com/tpapp/LogDensityProblems.jl#master";
# isinstalled("DynamicHMC") || pkg"add https://github.com/tpapp/DynamicHMC.jl#master";
