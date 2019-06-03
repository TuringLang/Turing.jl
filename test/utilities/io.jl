using Turing, Random, Test

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
end
