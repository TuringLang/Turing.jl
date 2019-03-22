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

# Wrapper function to check MoGtest.
function check_MoGtest_default(chain; eps = 0.2)
    check_numerical(chain,
        [:z1, :z2, :z3, :z4, :mu1, :mu2],
        [1.0, 1.0, 2.0, 2.0, 1.0, 4.0],
        eps=eps)
end

function test_ad(f, at = 0.5; rtol = 1e-8, atol = 1e-8)
    reverse = Tracker.gradient(f, at)[1]
    forward = ForwardDiff.derivative(f, at)
    finite_diff = central_fdm(5,1)(f, at)
    @test isapprox(reverse, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse, finite_diff, rtol=rtol, atol=atol)
end

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo()
    model(vi, SampleFromPrior())

    # Collect symbols.
    vnms = Array{Symbol}(undef, length(syms))
    vnvals = Array(undef, length(syms))
    for i in 1:length(syms)
        s = syms[i]
        vnms[i] = collect(
            Iterators.filter(vn -> vn.sym == s, keys(vi))
        )[1]
        vnvals[i] = getval(vi, vnms[i])[1]
    end


    spl = SampleFromPrior()
    _, ∇E = gradient_logp_forward(vi[spl], vi, model)
    grad_Turing = sort(∇E)

    # Call ForwardDiff's AD
    g = x -> ForwardDiff.gradient(f, x);
    grad_FWAD = sort(g(vnvals))

    # Compare result
    @test grad_Turing ≈ grad_FWAD atol=1e-9
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

varname(s::Symbol) = nothing, s
function varname(expr::Expr)
    # Initialize an expression block to store the code for creating uid
    local sym
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    indexing_ex = Expr(:block)
    # Add the initialization statement for uid
    push!(indexing_ex.args, quote indexing_list = [] end)
    # Initialize a local container for parsing and add the expr to it
    to_eval = []; pushfirst!(to_eval, expr)
    # Parse the expression and creating the code for creating uid
    find_head = false
    while length(to_eval) > 0
        evaling = popfirst!(to_eval)   # get the current expression to deal with
        if isa(evaling, Expr) && evaling.head == :ref && ~find_head
            # Add all the indexing arguments to the left
            pushfirst!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
            # Add first argument depending on its type
            # If it is an expression, it means it's a nested array calling
            # Otherwise it's the symbol for the calling
            if isa(evaling.args[1], Expr)
                pushfirst!(to_eval, evaling.args[1])
            else
                # push!(indexing_ex.args, quote pushfirst!(indexing_list, $(string(evaling.args[1]))) end)
                push!(indexing_ex.args, quote sym = Symbol($(string(evaling.args[1]))) end) # store symbol in runtime
                find_head = true
                sym = evaling.args[1] # store symbol in compilation time
            end
        else
            # Evaluting the concrete value of the indexing variable
            push!(indexing_ex.args, quote push!(indexing_list, string($evaling)) end)
        end
    end
    push!(indexing_ex.args, quote indexing = reduce(*, indexing_list) end)
    return indexing_ex, sym
end

genvn(sym::Symbol) = VarName(gensym(), sym, "", 1)
function genvn(expr::Expr)
    ex, sym = varname(expr)
    VarName(gensym(), sym, eval(ex), 1)
end

randr(vi::VarInfo,
    vn::VarName,
    dist::Distribution,
    spl::Turing.Sampler) = begin
    if ~haskey(vi, vn)
        r = rand(dist)
        Turing.push!(vi, vn, r, dist, spl.selector)
        spl.info[:cache_updated] = CACHERESET
        r
    elseif is_flagged(vi, vn, "del")
        unset_flag!(vi, vn, "del")
        r = rand(dist)
        vi[vn] = Turing.vectorize(dist, r)
        Turing.setorder!(vi, vn, vi.num_produce)
        r
    else
        Turing.updategid!(vi, vn, spl)
        vi[vn]
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
