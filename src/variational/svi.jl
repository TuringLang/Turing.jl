struct SVI{AD, A, D, F} <: VariationalInference{AD} where {A <: VariationalInference{AD}, D <: NamedTuple, F}
    alg::A
    data::D
    batch::F
    n::Int
end
function SVI(alg::A, data::D, batch::F, n::Int) where {AD, A <: VariationalInference{AD}, D, F}
    return SVI{AD, A, D, F}(alg, data, batch, n)
end

SVI(alg, data, batch) = SVI(alg, data, batch, size(first(data))[end])

alg_str(alg::SVI) = "SVI{$(alg_str(alg.alg))}"

function optimize!(
    vo,
    svi::SVI,
    q::VariationalPosterior,
    model::Model,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad()
)
    alg_name = alg_str(svi)
    samples_per_step = svi.alg.samples_per_step
    max_iters = svi.alg.max_iters

    s = first(keys(svi.data))  # one of the symbols to be changed in the batch
    n = svi.n # total number of samples
    batch_size = size(model.data[s])[end]

    # total number of iterations needed to get through a full dataset once
    total_iters = Integer(floor(n / batch_size)) * max_iters

    # fail we're not going through the entire dataset
    a = length(svi.data[s]) / length(model.data[s])
    if !((a - Integer(floor(a))) == 0.0)
        @warn "number of samples $n is not an integer multiple of $batch_size"
    end

    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (θ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$alg_name] Should only be seen once: optimizer created for θ" objectid(θ)
    end

    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(total_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged

        for batch in svi.batch(svi.data, batch_size)
            update_data!(model, batch)
            grad!(vo, svi.alg, q, model, θ, diff_result, samples_per_step, n / length(batch))

            # apply update rule
            Δ = DiffResults.gradient(diff_result)
            Δ = apply!(optimizer, θ, Δ)
            @. θ = θ - Δ

            PROGRESS[] && (ProgressMeter.next!(prog))
        end

        Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)

        i += 1
    end

    return θ
end

function (elbo::ELBO)(
    svi::SVI,
    q::TransformedDistribution{<:TuringDiagNormal},
    model::Model,
    θ::AbstractVector{T},
    num_samples
) where T <: Real
    # get the batch-size
    s = first(keys(svi.data))
    n = svi.n
    batch_size = size(model.data[s])[end]

    # weight to ensure we're only adding 1 × entropy(q) rather than n × entropy(q)
    w = n / batch_size

    elbo_acc = 0.0

    for batch in svi.batch(svi.data, batch_size)
        update_data!(model, batch)
        elbo_acc += elbo(svi.alg, q, model, θ, num_samples, w) / w
    end

    return elbo_acc
end


# TODO: make it generated or something
function update_data!(model::Model, data::NamedTuple)
    for s in keys(data)
        sym = Symbol(s)
        model.data[sym] .= data[sym]
    end

    return model
end
