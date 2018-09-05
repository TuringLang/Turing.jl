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

# test 
