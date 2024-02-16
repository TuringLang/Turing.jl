using Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD
using Turing.Bijectors

function test_enzyme(model::DynamicPPL.Model; linked=true)
    f = DynamicPPL.LogDensityFunction(model);
    if linked
        # This "link" the variables, i.e. include the transformation
        # from unconstrained to constrained space. This is what is used
        # by samplers.
        DynamicPPL.link!!(f.varinfo, model)
    end
    f_with_grad = LogDensityProblemsAD.ADgradient(:Enzyme, f);
    return LogDensityProblems.logdensity_and_gradient(f_with_grad, f.varinfo[:])
end

@model function hmcmatrixsup()
    v ~ Wishart(7, [1 0.5; 0.5 1])
end

_, y = Enzyme.autodiff(Enzyme.ReverseWithPrimal, logdensity, Enzyme.Active,
Enzyme.Const(ℓ), Enzyme.Duplicated(x, ∂ℓ_∂x))

model = hmcmatrixsup()
@run test_enzyme(model)


dist = Wishart(7, [1 0.5; 0.5 1])

dist_unconstrained = transformed(dist)

x = rand(dist_unconstrained)

# LogDensityProblems.jl interface for Wishart.
logp(x) = logpdf(transformed(Wishart(7, [1 0.5; 0.5 1])), x)
LogDensityProblems.dimension(::typeof(logp)) = 3
LogDensityProblems.logdensity(::typeof(logp), x) = logp(x)
LogDensityProblems.capabilities(::typeof(logp)) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.logdensity(logp, x)

logp_with_grad = LogDensityProblemsAD.ADgradient(:Enzyme, logp)
LogDensityProblems.logdensity_and_gradient(logp_with_grad, x)

ℓ = logp_with_grad.ℓ
∂ℓ_∂x = zero(x)

Enzyme.Const(ℓ)
Enzyme.Duplicated(x, ∂ℓ_∂x)

@run Enzyme.autodiff(Enzyme.ReverseWithPrimal, logp, Enzyme.Active, Enzyme.Duplicated(x, ∂ℓ_∂x))

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ReverseMode},
    x::AbstractVector)
    @unpack ℓ = ∇ℓ
    ∂ℓ_∂x = zero(x)
    _, y = Enzyme.autodiff(Enzyme.ReverseWithPrimal, logdensity, Enzyme.Active,
    Enzyme.Const(ℓ), Enzyme.Duplicated(x, ∂ℓ_∂x))
    y, ∂ℓ_∂x
end

@enter logpdf(transformed(Wishart(7, [1 0.5; 0.5 1])), x)

using LinearAlgebra

A = [4.0 12.0; 12.0 37.0]
LinearAlgebra.LAPACK.potrf!('U', deepcopy(A))

_potrf(A) = LinearAlgebra.LAPACK.potrf!('U', deepcopy(A))[1] 

∂ℓ_∂x = zeros(size(A)...)
result, grad = Enzyme.autodiff(Enzyme.ReverseWithPrimal, _potrf, Enzyme.Active, Enzyme.Duplicated(A, ∂ℓ_∂x))

Base.Fix1(LinearAlgebra.LAPACK.potrf!, 'U')(A)
# https://github.com/EnzymeAD/Enzyme.jl/issues/1081
@enter LinearAlgebra.LAPACK.potrf!('U', deepcopy(A))


@model function demo_hmc_prior()
    # NOTE: Used to use `InverseGamma(2, 3)` but this has infinite variance
    # which means that it's _very_ difficult to find a good tolerance in the test below:)
    s ~ truncated(Normal(3, 1), lower=0)
    m ~ Normal(0, sqrt(s))
end
gdemo_default_prior = DynamicPPL.contextualize(demo_hmc_prior(), DynamicPPL.PriorContext())

test_enzyme(gdemo_default_prior)


using Pkg
Pkg.activate(; temp=true)
Pkg.develop(path="/home/sunxd/Enzyme.jl")
Pkg.add("LogDensityProblems")
Pkg.add("LogDensityProblemsAD")
Pkg.add("Turing")

using Turing, Enzyme, LogDensityProblems, LogDensityProblemsAD
using Turing.Bijectors

dist = Wishart(7, [1 0.5; 0.5 1])
dist_unconstrained = transformed(dist)
x = rand(dist_unconstrained)
# LogDensityProblems.jl interface for Wishart.
logp(x) = logpdf(transformed(Wishart(7, [1 0.5; 0.5 1])), x)
LogDensityProblems.dimension(::typeof(logp)) = 3
LogDensityProblems.logdensity(::typeof(logp), x) = logp(x)
LogDensityProblems.capabilities(::typeof(logp)) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.logdensity(logp, x)

logp_with_grad = LogDensityProblemsAD.ADgradient(:Enzyme, logp)
@run LogDensityProblems.logdensity_and_gradient(logp_with_grad, x)

ℓ = logp_with_grad.ℓ
∂ℓ_∂x = zero(x)

Enzyme.Const(ℓ)
Enzyme.Duplicated(x, ∂ℓ_∂x)

@run Enzyme.autodiff(Enzyme.ReverseWithPrimal, logp, Enzyme.Active, Enzyme.Duplicated(x, ∂ℓ_∂x))

using LinearAlgebra
using LinearAlgebra: @blasname, lib

function EnzymeRules.forward(
    ::Const{typeof()}
)