using Turing
using DynamicPPL: DynamicPPL
using DynamicPPL.TestUtils.AD: run_ad
using ADTypes: AutoEnzyme
using Test: @test, @testset
using StableRNGs: StableRNG
import Enzyme: set_runtime_activity, Forward, Reverse, Const
import ForwardDiff  # needed for AD correctness checking

ADTYPES = (
    (
        "EnzymeForward",
        AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    ),
    (
        "EnzymeReverse",
        AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
    ),
)
MODELS = DynamicPPL.TestUtils.DEMO_MODELS

@testset verbose = true "AD / GibbsContext" begin
    @testset "adtype=$adtype_name" for (adtype_name, adtype) in ADTYPES
        @testset "model=$(model.f)" for model in MODELS
            global_vi = DynamicPPL.VarInfo(model)
            @testset for varnames in ([@varname(s)], [@varname(m)])
                @info "Testing Gibbs AD with adtype=$(adtype_name), model=$(model.f), varnames=$varnames"
                conditioned_model = Turing.Inference.make_conditional(
                    model, varnames, deepcopy(global_vi)
                )
                @test run_ad(
                    model, adtype; rng=StableRNG(468), test=true, benchmark=false
                ) isa Any
            end
        end
    end
end

@testset verbose = true "AD / Gibbs sampling" begin
    @testset "adtype=$adtype_name" for (adtype_name, adtype) in ADTYPES
        spl = Gibbs(
            @varname(s) => HMC(0.1, 10; adtype=adtype),
            @varname(m) => HMC(0.1, 10; adtype=adtype),
        )
        @testset "model=$(model.f)" for model in MODELS
            @info "Testing Gibbs sampling with adtype=$adtype_name, model=$(model.f)"
            @test sample(StableRNG(468), model, spl, 2; progress=false) isa Any
        end
    end
end
