module InitialParamsTests

using Distributions: Normal, Uniform
using DynamicPPL: DynamicPPL
import Random
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset, @test_logs
using Turing

@testset verbose = true "Testing initial_params.jl" begin
    @info "Starting initial parameter retry logic tests"
    
    @testset "External sampler with difficult initialization" begin
        # Model that produces -Inf logp for most parameter values
        # Only valid when x is in narrow range [-0.3, 0.3]
        @model function bad_init_model()
            x ~ Normal(0, 10)  # Prior is wide, so most samples will be outside valid range
            # Add log probability that's -Inf outside narrow range
            Turing.@addlogprob! (abs(x) < 0.3) ? 0.0 : -Inf
        end
        
        # This should succeed with retry logic (might take a few attempts)
        # Using external sampler that requires gradients
        rng = StableRNG(123)
        model = bad_init_model()
        
        # Test with NUTS (internal HMC sampler with retry logic)
        @testset "NUTS sampler" begin
            chain = sample(rng, model, NUTS(0.65), 10)
            @test size(chain, 1) == 10
            # Check that samples are in valid range
            x_samples = chain[:x]
            @test all(abs.(x_samples) .< 0.3)
        end
        
        # Test with HMC (should also work with retry logic)
        @testset "HMC sampler" begin
            chain = sample(StableRNG(456), model, HMC(0.1, 5), 10)
            @test size(chain, 1) == 10
            x_samples = chain[:x]
            @test all(abs.(x_samples) .< 0.3)
        end
    end
    
    @testset "Model with very narrow valid region" begin
        # Even more restrictive model - valid only in tiny range
        @model function very_bad_init_model()
            x ~ Normal(0, 100)  # Very wide prior
            # Valid only in range [-0.1, 0.1]
            Turing.@addlogprob! (abs(x) < 0.1) ? 0.0 : -Inf
        end
        
        rng = StableRNG(789)
        model = very_bad_init_model()
        
        # Should eventually find valid parameters
        chain = sample(rng, model, NUTS(0.65), 10)
        @test size(chain, 1) == 10
        x_samples = chain[:x]
        @test all(abs.(x_samples) .< 0.1)
    end
    
    @testset "Model that always fails initialization" begin
        # Model that's impossible to initialize (always -Inf)
        @model function impossible_model()
            x ~ Normal(0, 1)
            Turing.@addlogprob! -Inf  # Always invalid
        end
        
        model = impossible_model()
        
        # Should throw an error with informative message
        @test_throws ErrorException sample(StableRNG(999), model, NUTS(0.65), 10)
        
        # Check that error message is informative
        try
            sample(StableRNG(999), model, NUTS(0.65), 10)
        catch e
            error_msg = sprint(showerror, e)
            @test occursin("Failed to find valid initial parameters", error_msg)
            @test occursin("attempts", error_msg)
        end
    end
    
    @testset "Warning at attempt 10" begin
        # Model that requires many attempts
        @model function difficult_model()
            x ~ Normal(0, 50)
            # Valid only in tiny range, should trigger warning
            Turing.@addlogprob! (abs(x) < 0.05) ? 0.0 : -Inf
        end
        
        model = difficult_model()
        
        # Should see warning at attempt 10
        @test_logs(
            (:warn, r"failed to find valid initial parameters in 10 tries"),
            match_mode=:any,
            sample(StableRNG(111), model, NUTS(0.65), 10)
        )
    end
    
    @testset "Normal model initialization" begin
        # Standard model that should initialize easily
        @model function easy_model()
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end
        
        model = easy_model()
        
        # Should work without any retries or warnings
        chain = sample(StableRNG(222), model, NUTS(0.65), 10)
        @test size(chain, 1) == 10
    end
end

end  # module