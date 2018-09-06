using Turing, Distributions
using Test

mtest = quote gdemo2(x; a = 2, b = 2) = begin
		s ~ InverseGamma(a,b)
		m ~ Normal(0,sqrt.(s))
		for i = 1:length(x)
			x[i] ~ Normal(m, sqrt.(s))
		end
		return(s, m, x)
	end
end

fexpr = Turing.translate(mtest)
@info(fexpr)

# test extraction of function name, arguments and parameters
fname, fargs, fbody = Turing.extractcomponents(fexpr)

@test fname == :gdemo2
@test fargs[1].head == :parameters
@test :x in fargs
@test length(fargs) == 2
@test fbody == fexpr.args[2].args[2]

# test the insertion of the VarInfo statements
fbody2 = Turing.insertvarinfo(fbody)

@test fbody2.args[1] == :(_lp = zero(Real))
@test fbody2.args[end-1] == :(vi.logp = _lp)
@test fbody2.args[end] == :(return vi)

# test function construction
ftest = Turing.constructfunc(:test_f, [:x, :y], Expr(:block, Expr(:return, 1)))
eval(ftest)

@test test_f(1, 1) == 1
