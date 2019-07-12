using Turing, Random, Test
using Turing: Utilities

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "io.jl" begin
    @testset "chain save/resume" begin
        Random.seed!(1234)

        alg1 = HMCDA(3000, 1000, 0.65, 0.15)
        alg2 = PG(20, 500)
        alg3 = Gibbs(500, PG(30, 10, :s), HMCDA(1, 500, 0.65, 0.05, :m))

        chn1 = sample(gdemo_default, alg1; save_state=true)
        check_gdemo(chn1)

        chn1_resumed = Turing.Inference.resume(chn1, 1000)
        check_gdemo(chn1_resumed)

        chn1_contd = sample(gdemo_default, alg1; resume_from=chn1)
        check_gdemo(chn1_contd)

        chn1_contd2 = sample(gdemo_default, alg1; resume_from=chn1, reuse_spl_n=1000)
        check_gdemo(chn1_contd2)

        chn2 = sample(gdemo_default, alg2; save_state=true)
        check_gdemo(chn2)

        chn2_contd = sample(gdemo_default, alg2; resume_from=chn2)
        check_gdemo(chn2_contd)

        chn3 = sample(gdemo_default, alg3; save_state=true)
        check_gdemo(chn3)

        chn3_contd = sample(gdemo_default, alg3; resume_from=chn3)
        check_gdemo(chn3_contd)
    end
    @testset "split var string" begin
        var_str = "x"
        sym, inds = Utilities.split_var_str(var_str)
        @test sym == "x"
        @test inds == Vector{String}[]

        var_str = "x[1,1][2,3]"
        sym, inds = Utilities.split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["1", "1"]
        @test inds[2] == ["2", "3"]

        var_str = "x[Colon(),1][2,Colon()]"
        sym, inds = Utilities.split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["Colon()", "1"]
        @test inds[2] == ["2", "Colon()"]

        var_str = "x[2:3,1][2,1:2]"
        sym, inds = Utilities.split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["2:3", "1"]
        @test inds[2] == ["2", "1:2"]

        var_str = "x[2:3,2:3][[1,2],[1,2]]"
        sym, inds = Utilities.split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["2:3", "2:3"]
        @test inds[2] == ["[1,2]", "[1,2]"]
    end
end
