using Test
using Turing
using Random

BATCH_SIZE = 2

# full dataset
data = (x = [1.0, 2.0, 3.0, 4.0, 5.0], )  # NamedTuple containing the full dataset
data = (x = randn(1000), )

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
advi = ADVI(1, 1000)

# Wrap in SVI to use mini-batches for gradient estimation, ELBO, etc.
svi = Turing.SVI(advi, data, batch_gen)

@model gdemo(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, √s)

    n = length(x)
    x ~ MvNormal(m * ones(n), √s)
end

m = gdemo(first(svi.batch_gen(svi.data, BATCH_SIZE))...)

q = vi(m, svi)

using DynamicPPL
@model gdemo_with_missing(x, y = missing) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, √s)

    n = length(x)
    x ~ MvNormal(m * ones(n), √s)
    y ~ Normal(0, 1)
end


data = (x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], )

m = gdemo(data...)
m()

data = get_data(m)


# Generating batches
using Base.Iterators
initial_batch = first(svi.batch_gen(svi.data, 2))

gen = Base.Iterators.Stateful(svi.batch_gen(svi.data, 2))
first(gen) # drop first entry
batch = first(gen) # second entry


# Testing a batch
m = gdemo(x = [1.0, 2.0])
batch_size = size(first(Turing.Variational.get_data(m)))[end]

logπ = Turing.Variational.make_logjoint(m; weight = 3.)

mdata_old = deepcopy(get_data(m))
logπ_old = logπ([1.0, 0.0])

Turing.Variational.update_data!(m, (x = [3.0, 4.0], ))

logπ_new = logπ([1.0, 0.0])
mdata_new = deepcopy(get_data(m))

@test mdata_new.x ≠ mdata_old.x
@test logπ_new ≠ logπ_old
