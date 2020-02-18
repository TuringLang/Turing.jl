"""
    SVI(alg::VariationalInference, data, batch_gen)
    SVI(alg::VariationalInference, data, batch_gen, n)

Wraps the given VI algorithm `alg` to perform computations batch-wise rather than
using the full dataset.

When using mini-batches for gradient estimation rather than the entire dataset,
the method is sometimes referred to as _Stochastic Variational Inference (SVI)_.[1, 2]

This will also, when possible, compute whatever objective batch-wise. For example,
if one runs `elbo(svi, ...)` rather than `elbo(advi, ...)` we will compute the
ELBO on a per-batch basis and sum together to get the ELBO for the entire dataset.
Worth noting that this may have reduced numerical accuracy compared to evaluating
on the entire dataset in one go. Therefore it's advised to use a model instantiated
with the full dataset when computing the objective, if possible.

# Arguments
- `alg::VariationalInference`: the VI algorithm to wrap
- `data`: usually a `NamedTuple` with keys corresponding to the observed argument
  in the `Model` definition, but can be any data source as long as it's handled
  correctly in `batch_gen` (see below). It could for example be a list of files
  to lazy-load the data from.
- `batch_gen`: a callable with signature `batch_gen(data, batch_size)::iterable`
  where the iterator yields `NamedTuple` with keys corresponding to the observed
  arguments in the `Model` definition
- `n::Int`: the number of samples in the full dataset. If no `n` is provided,
  `size(first(data))[end]` will be used. This assumes two things:
  1. First entry in `data` has `size` defined.
  2. The last dimension is hte batch-dimension.
  Either way, it's required to know the number of samples beforehand as certain
  objectives require re-weighting when subsampling.

# Examples
## ADVI with mini-batches
One classical example is to simply use stochastic gradient descent (SGD) with
`ADVI`, i.e. gradient estimates are computed using subsets of the full dataset.

```julia
BATCH_SIZE = 2

# full dataset
data = (x = [1.0, 2.0, 3.0, 4.0, 5.0], )  # NamedTuple containing the full dataset

# constructor for the batch generator
function batch_gen(data, batch_size)
    # total number of datapoints
    n = length(data.x)

    # shuffle the dataset indices
    indices = shuffle(1:n)

    return (
        (x = data.x[indices[i:i - 1 + batch_size]], )
        for i = 1:batch_size:(n - batch_size + 1)
    )
end

# ADVI with 1 sample for gradient estimation and 10 total iterations through the full data
advi = ADVI(1, 10)

# Wrap in SVI to use mini-batches for gradient estimation, ELBO, etc.
svi = SVI(advi, data, batch_gen)

# initialize a model with an example of a batch to get the sizes right
example_batch = first(batch_gen(data, BATCH_SIZE))
m = model(example_batch...)

# perform VI as usual
q = vi(m, svi)
```

We can also evaluate the `elbo` batch-wise after fitting:
```julia
# compute ELBO using 1000 samples
elbo(svi, q, m, 1000)
```

# References
[1] Hoffman, M., Blei, D. M., Wang, C., & Paisley, J., Stochastic Variational Inference, CoRR, (),  (2012). 
[2] Zhang, C., Butepage, J., Kjellstrom, H., & Mandt, S., Advances in variational inference, CoRR, (),  (2017). 
"""
struct SVI{AD, A, D, F} <: VariationalInference{AD} where {A <: VariationalInference{AD}, D, F}
    alg::A
    data::D
    batch_gen::F
    n::Int
end
function SVI(alg::A, data::D, batch_gen::F, n::Int) where {AD, A <: VariationalInference{AD}, D, F}
    return SVI{AD, A, D, F}(alg, data, batch_gen, n)
end

SVI(alg, data, batch_gen) = SVI(alg, data, batch_gen, size(first(data))[end])

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

    s = first(keys(svi.data)) # one of the symbols to be changed in the batch
    n = svi.n                 # total number of samples
    batch_size = size(get_data(model)[s])[end]

    # total number of iterations needed to get through a full dataset once
    total_iters = Integer(floor(n / batch_size)) * max_iters

    # fail we're not going through the entire dataset
    a = length(svi.n) / length(batch_size)
    if !((a - Integer(floor(a))) == 0.0)
        @warn "number of samples $n is not an integer multiple of $batch_size"
    end

    logπ = make_logjoint(model; weight = n / batch_size)

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

        for batch in svi.batch_gen(svi.data, batch_size)
            update_data!(model, batch)
            grad!(vo, svi.alg, q, logπ, θ, diff_result, samples_per_step)

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

data_from_args(::NamedTuple{names}, ::Val{vals}) where {names, vals} = setdiff(names, vals)
function get_data(m::Model)
    data_keys = data_from_args(m.args, getmissing(m))
    return (; zip(data_keys, (m.args[k] for k in data_keys))...)
end

function update_data!(model::Model, data::NamedTuple{names}) where {names}
    # Extract the data from the model
    mdata = get_data(model)
    for s in names
        # Inplace updates to the model-data
        mdata[s] .= data[s]
    end

    return model
end
