
import DistributionsAD
export ADVI

Base.@deprecate meanfield(model) q_meanfield_gaussian(model)

struct ADVI{AD}
    "Number of samples used to estimate the ELBO in each optimization step."
    samples_per_step::Int
    "Maximum number of gradient steps."
    max_iters::Int
    "AD backend used for automatic differentiation."
    adtype::AD
end

function ADVI(
    samples_per_step::Int=1,
    max_iters::Int=1000;
    adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
)
    Base.depwarn(
        "The type ADVI will be removed in future releases. Please refer to the new interface for `vi`",
        :ADVI;
        force=true,
    )
    return ADVI{typeof(adtype)}(samples_per_step, max_iters, adtype)
end

function vi(model::DynamicPPL.Model, alg::ADVI; kwargs...)
    Base.depwarn(
        "This specialization along with the type `ADVI`  will be deprecated in future releases. Please refer to the new interface for `vi`.",
        :vi;
        force=true,
    )
    q = q_meanfield_gaussian(Random.default_rng(), model)
    objective = AdvancedVI.RepGradELBO(
        alg.samples_per_step; entropy=AdvancedVI.ClosedFormEntropy()
    )
    operator = AdvancedVI.IdentityOperator()
    _, q_avg, _, _ = vi(model, q, alg.max_iters; objective, operator, kwargs...)
    return q_avg
end

function vi(
    model::DynamicPPL.Model,
    alg::ADVI,
    q::Bijectors.TransformedDistribution{<:DistributionsAD.TuringDiagMvNormal};
    kwargs...,
)
    Base.depwarn(
        "This specialization along with the type `ADVI`  will be deprecated in future releases. Please refer to the new interface for `vi`.",
        :vi;
        force=true,
    )
    objective = AdvancedVI.RepGradELBO(
        alg.samples_per_step; entropy=AdvancedVI.ClosedFormEntropy()
    )
    operator = AdvancedVI.IdentityOperator()
    _, q_avg, _, _ = vi(model, q, alg.max_iters; objective, operator, kwargs...)
    return q_avg
end
