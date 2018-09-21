using DynamicHMC, ForwardDiff, Distributions, LogDensityProblems
using LogDensityProblems: AbstractLogDensityProblem, ValueGradient
using Turing, LinearAlgebra
using ForwardDiff

struct FunctionLogDensity{F} <: AbstractLogDensityProblem
  dimension::Int
  f::F
end

LogDensityProblems.dimension(ℓ::FunctionLogDensity) = ℓ.dimension

LogDensityProblems.logdensity(::Type{ValueGradient}, ℓ::FunctionLogDensity, x) =
    ℓ.f(x)::ValueGradient


# Sample a precision matrix A from a Wishart distribution
# with identity scale matrix and 250 degrees of freedome
dim2 = 25
A   = rand(Wishart(dim2, Matrix{Float64}(I, dim2, dim2)))
d   = MvNormal(zeros(dim2), A)

function lp(x)
    value = logpdf(d, x)
    deriv = ForwardDiff.gradient(x->logpdf(d, x), x)
    return ValueGradient(value, deriv)
end

lpp = FunctionLogDensity(dim2, lp)


n_iter, n_name, n_chain = 1000, dim2, 1
chain, NUTS_tuned = NUTS_init_tune_mcmc(lpp, n_iter)

samples = get_position.(chain)
mean(samples)



val = randn(n_iter, n_name, n_chain) .+ [1:dim2...]';
val = hcat(val, samples);

# construct a Chains object
chn = Chains(val);
