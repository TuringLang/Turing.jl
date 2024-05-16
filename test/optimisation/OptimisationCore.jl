# Used for testing how well it works with nested contexts.
struct OverrideContext{C,T1,T2} <: DynamicPPL.AbstractContext
    context::C
    logprior_weight::T1
    loglikelihood_weight::T2
end
DynamicPPL.NodeTrait(::OverrideContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(parent::OverrideContext) = parent.context
DynamicPPL.setchildcontext(parent::OverrideContext, child) = OverrideContext(
    child,
    parent.logprior_weight,
    parent.loglikelihood_weight
)

# Only implement what we need for the models above.
function DynamicPPL.tilde_assume(context::OverrideContext, right, vn, vi)
    value, logp, vi = DynamicPPL.tilde_assume(context.context, right, vn, vi)
    return value, context.logprior_weight, vi
end
function DynamicPPL.tilde_observe(context::OverrideContext, right, left, vi)
    logp, vi = DynamicPPL.tilde_observe(context.context, right, left, vi)
    return context.loglikelihood_weight, vi
end

@testset "OptimisationCore.jl" begin
    # Issue: https://discourse.julialang.org/t/two-equivalent-conditioning-syntaxes-giving-different-likelihood-values/100320
    @testset "OptimizationContext" begin
        @model function model1(x)
            μ ~ Uniform(0, 2)
            x ~ LogNormal(μ, 1)
        end

        @model function model2()
            μ ~ Uniform(0, 2)
            x ~ LogNormal(μ, 1)
        end

        x = 1.0
        w = [1.0]

        @testset "With ConditionContext" begin
            m1 = model1(x)
            m2 = model2() | (x=x,)
            ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end

        @testset "With prefixes" begin
            function prefix_μ(model)
                return DynamicPPL.contextualize(
                    model, DynamicPPL.PrefixContext{:inner}(model.context)
                )
            end
            m1 = prefix_μ(model1(x))
            m2 = prefix_μ(model2() | (var"inner.x"=x,))
            ctx = Turing.OptimizationContext(DynamicPPL.LikelihoodContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end

        @testset "Weighted" begin
            function override(model)
                return DynamicPPL.contextualize(
                    model,
                    OverrideContext(model.context, 100, 1)
                )
            end
            m1 = override(model1(x))
            m2 = override(model2() | (x=x,))
            ctx = Turing.OptimizationContext(DynamicPPL.DefaultContext())
            @test Turing.OptimLogDensity(m1, ctx)(w) == Turing.OptimLogDensity(m2, ctx)(w)
        end
    end
end
