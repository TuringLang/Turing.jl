using Test
using Turing
using Random

@testset "Minibatch VI" begin
    BATCH_SIZE = 2

    # full dataset
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

    # Wrap in MinibatchVI to use mini-batches for gradient estimation, ELBO, etc.
    svi = Turing.MinibatchVI(advi, data, batch_gen)

    @model gdemo2(x) = begin
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)

        n = length(x)
        x ~ MvNormal(m * ones(n), √s)
    end

    @testset "Minibatch ADVI vs. ADVI on gdemo" begin
        # all data
        m = gdemo2(svi.data...)
        q = vi(m, advi)
        zs = rand(q, 1000)
        z_mean = mean(zs; dims = 2)

        # subsample data
        m = gdemo2(first(svi.batch_gen(svi.data, BATCH_SIZE))...)
        q_minibatch = vi(m, svi)
        zs = rand(q, 1000)
        z_mean_minibatch = mean(zs; dims = 2)

        @test sum(abs2, z_mean - z_mean_minibatch) ≤ 1e-3
    end


    @testset "get_data with `missing`" begin
        using DynamicPPL
        @model gdemo_with_missing(x, y = missing) = begin
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, √s)

            n = length(x)
            x ~ MvNormal(m * ones(n), √s)
            y ~ Normal(0, 1)
        end


        data = (x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], )

        m = gdemo_with_missing(x = data.x)
        data = Turing.Variational.get_data(m)

        @test :y ∉ keys(data)
    end


    @testset "Batch generation" begin
        using Base.Iterators
        initial_batch = first(svi.batch_gen(svi.data, 2))

        gen = Base.Iterators.Stateful(svi.batch_gen(svi.data, 2))
        batch1 = first(gen) # drop first entry
        batch2 = first(gen) # second entry

        @test batch1.x ≠ batch2.x
    end

    # Testing a batch
    @testset "Inplace update of data" begin
        m = gdemo2(x = [1.0, 2.0])
        batch_size = size(first(Turing.Variational.get_data(m)))[end]

        logπ = Turing.Variational.make_logjoint(m; weight = 3.)

        mdata_old = deepcopy(Turing.Variational.get_data(m))
        logπ_old = logπ([1.0, 0.0])

        Turing.Variational.update_data!(m, (x = [3.0, 4.0], ))

        logπ_new = logπ([1.0, 0.0])
        mdata_new = deepcopy(Turing.Variational.get_data(m))

        @test mdata_new.x ≠ mdata_old.x
        @test logπ_new ≠ logπ_old
    end
end
