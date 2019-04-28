using Turing, Random
using Turing: Selector, reconstruct, invlink, CACHERESET, 
    SampleFromPrior, Sampler, runmodel!
using Turing.RandomVariables
using Turing.RandomVariables: uid, cuid, getvals, getidcs,
    set_retained_vns_del_by_spl!, is_flagged, 
    set_flag!, unset_flag!, is_inside
using Distributions
using ForwardDiff: Dual
using Test

i, j, k = 1, 2, 3

include("../test_utils/AllUtils.jl")

@testset "RandomVariables.jl" begin
    @turing_testset "runmodel!" begin
        @model testmodel() = begin
            x ~ Normal()
        end
        alg = HMC(1000, 0.1, 5)
        spl = Sampler(alg)
        vi = VarInfo()
        m = testmodel()
        m(vi)
        runmodel!(m, vi, spl)
        @test spl.info[:eval_num] == 1
        runmodel!(m, vi, spl)
        @test spl.info[:eval_num] == 2
    end
    @turing_testset "flags" begin
        vi = VarInfo()
        vn_x = VarName(gensym(), :x, "", 1)
        dist = Normal(0, 1)
        r = rand(dist)
        gid = Selector()

        push!(vi, vn_x, r, dist, gid)

        # del is set by default
        @test is_flagged(vi, vn_x, "del") == false

        set_flag!(vi, vn_x, "del")
        @test is_flagged(vi, vn_x, "del") == true

        unset_flag!(vi, vn_x, "del")
        @test is_flagged(vi, vn_x, "del") == false
    end
    @turing_testset "is_inside" begin
        space = Set([:x, :y, :(z[1])])
        vn1 = genvn(:x)
        vn2 = genvn(:y)
        vn3 = genvn(:(x[1]))
        vn4 = genvn(:(z[1][1]))
        vn5 = genvn(:(z[2]))
        vn6 = genvn(:z)

        @test is_inside(vn1, space)
        @test is_inside(vn2, space)
        @test is_inside(vn3, space)
        @test is_inside(vn4, space)
        @test ~is_inside(vn5, space)
        @test ~is_inside(vn6, space)
    end
    @testset "orders" begin
        randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Turing.Sampler) = begin
            if ~haskey(vi, vn)
                r = rand(dist)
                Turing.push!(vi, vn, r, dist, spl.selector)
                spl.info[:cache_updated] = CACHERESET
                r
            elseif is_flagged(vi, vn, "del")
                unset_flag!(vi, vn, "del")
                r = rand(dist)
                vi[vn] = Turing.vectorize(dist, r)
                Turing.setorder!(vi, vn, vi.num_produce)
                r
            else
                Turing.updategid!(vi, vn, spl)
                vi[vn]
            end
        end

        csym = gensym() # unique per model
        vn_z1 = VarName(csym, :z, "[1]", 1)
        vn_z2 = VarName(csym, :z, "[2]", 1)
        vn_z3 = VarName(csym, :z, "[3]", 1)
        vn_z4 = VarName(csym, :z, "[4]", 1)
        vn_a1 = VarName(csym, :a, "[1]", 1)
        vn_a2 = VarName(csym, :a, "[2]", 1)
        vn_b = VarName(csym, :b, "", 1)

        vi = VarInfo()
        dists = [Categorical([0.7, 0.3]), Normal()]

        spl1 = Turing.Sampler(PG(5,5))
        spl2 = Turing.Sampler(PG(5,5))

        # First iteration, variables are added to vi
        # variables samples in order: z1,a1,z2,a2,z3
        vi.num_produce += 1
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        vi.num_produce += 1
        randr(vi, vn_b, dists[2], spl2)
        randr(vi, vn_z2, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        vi.num_produce += 1
        randr(vi, vn_z3, dists[1], spl1)
        @test vi.orders == [1, 1, 2, 2, 2, 3]
        @test vi.num_produce == 3

        vi.num_produce = 0
        set_retained_vns_del_by_spl!(vi, spl1)
        @test is_flagged(vi, vn_z1, "del")
        @test is_flagged(vi, vn_a1, "del")
        @test is_flagged(vi, vn_z2, "del")
        @test is_flagged(vi, vn_a2, "del")
        @test is_flagged(vi, vn_z3, "del")

        vi.num_produce += 1
        randr(vi, vn_z1, dists[1], spl1)
        randr(vi, vn_a1, dists[2], spl1)
        vi.num_produce += 1
        randr(vi, vn_z2, dists[1], spl1)
        vi.num_produce += 1
        randr(vi, vn_z3, dists[1], spl1)
        randr(vi, vn_a2, dists[2], spl1)
        @test vi.orders == [1, 1, 2, 2, 3, 3]
        @test vi.num_produce == 3
    end
    @turing_testset "replay" begin
        # Generate synthesised data
        xs = rand(Normal(0.5, 1), 100)

        # Define model
        @model priorsinarray(xs) = begin
            priors = Vector{Real}(undef, 2)
            priors[1] ~ InverseGamma(2, 3)
            priors[2] ~ Normal(0, sqrt(priors[1]))
            for i = 1:length(xs)
                xs[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        # Sampling
        chain = sample(priorsinarray(xs), HMC(10, 0.01, 10))
    end
    @turing_testset "test_varname" begin
        # Symbol
        v_sym = string(:x)
        @test v_sym == "x"

        # Array
        v_arr = eval(varname(:(x[i]))[1])
        @test v_arr == "[1]"

        # Matrix
        v_mat = eval(varname(:(x[i,j]))[1])
        @test v_mat== "[1,2]"

        v_mat = eval(varname(:(x[i,j,k]))[1])
        @test v_mat== "[1,2,3]"

        v_mat = eval(varname(:((x[1,2][1+5][45][3][i])))[1])
        @test v_mat == "[1,2][6][45][3][1]"

        @model mat_name_test() = begin
            p = Array{Any}(undef, 2, 2)
            for i in 1:2, j in 1:2
                p[i,j] ~ Normal(0, 1)
            end
            p
        end
        chain = sample(mat_name_test(), HMC(1000, 0.2, 4))
        check_numerical(chain, ["p[1, 1]"], [0], eps = 0.25)

        # Multi array
        v_arrarr = eval(varname(:(x[i][j]))[1])
        @test v_arrarr == "[1][2]"

        @model marr_name_test() = begin
            p = Array{Array{Any}}(undef, 2)
            p[1] = Array{Any}(undef, 2)
            p[2] = Array{Any}(undef, 2)
            for i in 1:2, j in 1:2
                p[i][j] ~ Normal(0, 1)
            end
            p
        end

        chain = sample(marr_name_test(), HMC(1000, 0.2, 4))
        check_numerical(chain, ["p[1][1]"], [0], eps = 0.25)
    end
    @turing_testset "varinfo" begin
        # Test for uid() (= string())
        csym = gensym()
        vn1 = VarName(csym, :x, "[1]", 1)
        @test string(vn1) == "{$csym,x[1]}:1"

        vn2 = VarName(csym, :x, "[1]", 2)
        vn11 = VarName(csym, :x, "[1]", 1)

        @test cuid(vn1) == cuid(vn2)
        @test vn11 == vn1

        vi = VarInfo()
        dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]

        spl2 = Turing.Sampler(PG(5,5))
        vn_w = VarName(gensym(), :w, "", 1)
        randr(vi, vn_w, dists[1], spl2, true)

        vn_x = VarName(gensym(), :x, "", 1)
        vn_y = VarName(gensym(), :y, "", 1)
        vn_z = VarName(gensym(), :z, "", 1)
        vns = [vn_x, vn_y, vn_z]

        spl1 = Turing.Sampler(PG(5,5))
        for i = 1:3
          r = randr(vi, vns[i], dists[i], spl1, false)
          val = vi[vns[i]]
          @test sum(val - r) <= 1e-9
        end

        @test length(getvals(vi, spl1)) == 3
        @test length(getvals(vi, spl2)) == 1

        vn_u = VarName(gensym(), :u, "", 1)
        randr(vi, vn_u, dists[1], spl2, true)

        vi.num_produce = 1
        set_retained_vns_del_by_spl!(vi, spl2)

        vals_of_1 = collect(getvals(vi, spl1))
        filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_1)
        @test length(vals_of_1) == 3

        vals_of_2 = collect(getvals(vi, spl2))
        filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_2)
        @test length(vals_of_2) == 1

        @model igtest() = begin
          x ~ InverseGamma(2,3)
          y ~ InverseGamma(2,3)
          z ~ InverseGamma(2,3)
          w ~ InverseGamma(2,3)
          u ~ InverseGamma(2,3)
        end

        # Test the update of group IDs
        g_demo_f = igtest()
        g = Turing.Sampler(Gibbs(1000, PG(10, 2, :x, :y, :z), HMC(1, 0.4, 8, :w, :u)), g_demo_f)

        pg, hmc = g.info[:samplers]

        vi = Turing.VarInfo()
        g_demo_f(vi, SampleFromPrior())
        vi, _ = Turing.Inference.step(g_demo_f, pg, vi)
        @test vi.gids == [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]),
                          Set{Selector}(), Set{Selector}()]

        g_demo_f(vi, hmc)
        @test vi.gids == [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]),
                          Set([hmc.selector]), Set([hmc.selector])]
    end
end
