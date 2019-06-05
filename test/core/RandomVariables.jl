using Turing, Random
using Turing: Selector, reconstruct, invlink, CACHERESET, 
    SampleFromPrior, Sampler, runmodel!, SampleFromUniform
using Turing.RandomVariables
using Turing.RandomVariables: uid, _getidcs,
    set_retained_vns_del_by_spl!, is_flagged, 
    set_flag!, unset_flag!, VarInfo, TypedVarInfo,
    getlogp, setlogp!, resetlogp!, acclogp!
using Distributions
using ForwardDiff: Dual
using Test

i, j, k = 1, 2, 3

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "RandomVariables.jl" begin
    @turing_testset "TypedVarInfo" begin
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2,3)
            m ~ TruncatedNormal(0.0,sqrt(s),0.0,2.0)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        vi = VarInfo()
        model(vi, SampleFromUniform())
        tvi = TypedVarInfo(vi)

        meta = vi.metadata
        for f in fieldnames(typeof(tvi.metadata))
            fmeta = getfield(tvi.metadata, f)
            for vn in fmeta.vns
                @test tvi[vn] == vi[vn]
                ind = meta.idcs[vn]
                tind = fmeta.idcs[vn]
                @test meta.dists[ind] == fmeta.dists[tind]
                @test meta.orders[ind] == fmeta.orders[tind]
                @test meta.gids[ind] == fmeta.gids[tind]
                for flag in keys(meta.flags)
                    @test meta.flags[flag][ind] == fmeta.flags[flag][tind]
                end
                range = meta.ranges[ind]
                trange = fmeta.ranges[tind]
                @test all(meta.vals[range] .== fmeta.vals[trange])
            end
        end
    end
    @turing_testset "Base" begin
        # Test Base functions:
        #   string, Symbol, ==, hash, in, keys, haskey, isempty, push!, empty!,
        #   getindex, setindex!, getproperty, setproperty!
        csym = gensym()
        vn1 = VarName(csym, :x, "[1][2]", 1)
        @test string(vn1) == "{$csym,x[1][2]}:1"
        @test string(vn1, all=false) == "x[1][2]"
        @test Symbol(vn1) == Symbol("x[1][2]")

        vn2 = VarName(csym, :x, "[1][2]", 1)
        @test vn2 == vn1
        @test hash(vn2) == hash(vn1)
        @test in(vn1, Set([:x]))

        function test_base!(vi)
            empty!(vi)
            @test vi.logp == 0
            @test vi.num_produce == 0

            vn = VarName(gensym(), :x, "", 1)
            dist = Normal(0, 1)
            r = rand(dist)
            gid = Selector()

            @test isempty(vi)
            @test ~haskey(vi, vn)
            push!(vi, vn, r, dist, gid)
            @test ~isempty(vi)
            @test haskey(vi, vn)

            @test length(vi[vn]) == 1
            @test length(vi[SampleFromPrior()]) == 1

            @test vi[vn] == r
            @test vi[SampleFromPrior()][1] == r
            vi[vn] = [2*r]
            @test vi[vn] == 2*r
            @test vi[SampleFromPrior()][1] == 2*r
            vi[SampleFromPrior()] = [3*r]
            @test vi[vn] == 3*r
            @test vi[SampleFromPrior()][1] == 3*r

            empty!(vi)
            @test isempty(vi)
            push!(vi, vn, r, dist, gid)

            function test_in()
                space = Set([:x, :y, :(z[1])])
                vn1 = genvn(:x)
                vn2 = genvn(:y)
                vn3 = genvn(:(x[1]))
                vn4 = genvn(:(z[1][1]))
                vn5 = genvn(:(z[2]))
                vn6 = genvn(:z)
    
                @test in(vn1, space)
                @test in(vn2, space)
                @test in(vn3, space)
                @test in(vn4, space)
                @test ~in(vn5, space)
                @test ~in(vn6, space)
            end
            test_in()
        end
        vi = VarInfo()
        test_base!(vi)
        test_base!(empty!(TypedVarInfo(vi)))
    end
    @turing_testset "runmodel!" begin
        # Test that eval_num is incremented when calling runmodel!
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
        # Test flag setting:
        #    is_flagged, set_flag!, unset_flag!
        function test_varinfo!(vi)
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
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!(TypedVarInfo(vi)))
    end
    @turing_testset "link!" begin
        # Test linking spl and vi:
        #    link!, invlink!, istrans
        @model gdemo(x, y) = begin
            s ~ InverseGamma(2,3)
            m ~ TruncatedNormal(0.0,sqrt(s),0.0,2.0)
            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))
        end
        model = gdemo(1.0, 2.0)

        vi = VarInfo()
        model(vi, SampleFromUniform())

        @test all(i->~istrans(vi, vi.vns[i]), 1:length(vi.vns))
        alg = HMC(1000, 0.1, 5)
        spl = Sampler(alg)
        v = copy(vi.vals)
        link!(vi, spl)
        @test all(i->istrans(vi, vi.vns[i]), 1:length(vi.vns))
        invlink!(vi, spl)
        @test all(i->~istrans(vi, vi.vns[i]), 1:length(vi.vns))
        @test vi.vals == v

        vi = TypedVarInfo(vi)
        alg = HMC(1000, 0.1, 5)
        spl = Sampler(alg)
        @test all(i->~istrans(vi, vi.metadata.s.vns[i]), 1:length(vi.metadata.s.vns))
        @test all(i->~istrans(vi, vi.metadata.m.vns[i]), 1:length(vi.metadata.m.vns))
        v_s = copy(vi.metadata.s.vals)
        v_m = copy(vi.metadata.m.vals)
        link!(vi, spl)
        @test all(i->istrans(vi, vi.metadata.s.vns[i]), 1:length(vi.metadata.s.vns))
        @test all(i->istrans(vi, vi.metadata.m.vns[i]), 1:length(vi.metadata.m.vns))
        invlink!(vi, spl)
        @test all(i->~istrans(vi, vi.metadata.s.vns[i]), 1:length(vi.metadata.s.vns))
        @test all(i->~istrans(vi, vi.metadata.m.vns[i]), 1:length(vi.metadata.m.vns))
        @test vi.metadata.s.vals == v_s
        @test vi.metadata.m.vals == v_m
    end
    @turing_testset "setgid!" begin
        vi = VarInfo()
        vn = VarName(gensym(), :x, "", 1)
        dist = Normal(0, 1)
        r = rand(dist)
        gid1 = Selector()
        gid2 = Selector(2, :HMC)

        push!(vi, vn, r, dist, gid1)
        @test vi.gids[vi.idcs[vn]] == Set([gid1])
        setgid!(vi, gid2, vn)
        @test vi.gids[vi.idcs[vn]] == Set([gid1, gid2])

        vi = empty!(TypedVarInfo(vi))
        push!(vi, vn, r, dist, gid1)
        @test vi.metadata.x.gids[vi.metadata.x.idcs[vn]] == Set([gid1])
        setgid!(vi, gid2, vn)
        @test vi.metadata.x.gids[vi.metadata.x.idcs[vn]] == Set([gid1, gid2])
    end
    @testset "orders" begin
        function randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Turing.Sampler)
            if ~haskey(vi, vn)
                r = rand(dist)
                Turing.push!(vi, vn, r, dist, spl)
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

        vi = empty!(TypedVarInfo(vi))
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
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 2]
        @test vi.metadata.b.orders == [2]
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
        @test vi.metadata.z.orders == [1, 2, 3]
        @test vi.metadata.a.orders == [1, 3]
        @test vi.metadata.b.orders == [2]
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
    @turing_testset "varname" begin
        csym = gensym()
        vn1 = VarName(csym, :x, "[1]", 1)
        @test vn1 == VarName{:x}(csym, "[1]", 1)
        @test vn1 == VarName{:x}(csym, "[1]")
        @test vn1 == VarName([csym, :x], "[1]")
        vn2 = VarName(csym, :x, "[2]", 1)
        @test vn2 == VarName(vn1, "[2]")

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
        dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]
        function test_varinfo!(vi)
            @test getlogp(vi) == 0
            setlogp!(vi, 1)
            @test getlogp(vi) == 1
            acclogp!(vi, 1)
            @test getlogp(vi) == 2
            resetlogp!(vi)
            @test getlogp(vi) == 0

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

            idcs = _getidcs(vi, spl1)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 3
            else
                @test length(idcs) == 3
            end
            @test length(vi[spl1]) == 7

            idcs = _getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 1
            else
                @test length(idcs) == 1
            end
            @test length(vi[spl2]) == 1

            vn_u = VarName(gensym(), :u, "", 1)
            randr(vi, vn_u, dists[1], spl2, true)

            idcs = _getidcs(vi, spl2)
            if idcs isa NamedTuple
                @test sum(length(getfield(idcs, f)) for f in fieldnames(typeof(idcs))) == 2
            else
                @test length(idcs) == 2
            end
            @test length(vi[spl2]) == 2
        end
        vi = VarInfo()
        test_varinfo!(vi)
        test_varinfo!(empty!(TypedVarInfo(vi)))

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
        vi = VarInfo()
        g_demo_f(vi, SampleFromPrior())
        vi, _ = Turing.Inference.step(g_demo_f, pg, vi)
        @test vi.gids == [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]),
                        Set{Selector}(), Set{Selector}()]

        g_demo_f(vi, hmc)
        @test vi.gids == [Set([pg.selector]), Set([pg.selector]), Set([pg.selector]),
                        Set([hmc.selector]), Set([hmc.selector])]

        g = Turing.Sampler(Gibbs(1000, PG(10, 2, :x, :y, :z), HMC(1, 0.4, 8, :w, :u)), g_demo_f)
        pg, hmc = g.info[:samplers]
        vi = empty!(TypedVarInfo(vi))
        g_demo_f(vi, SampleFromPrior())
        vi, _ = Turing.Inference.step(g_demo_f, pg, vi)
        g_demo_f(vi, hmc)
        @test vi.metadata.x.gids[1] == Set([pg.selector])
        @test vi.metadata.y.gids[1] == Set([pg.selector])
        @test vi.metadata.z.gids[1] == Set([pg.selector])
        @test vi.metadata.w.gids[1] == Set([hmc.selector])
        @test vi.metadata.u.gids[1] == Set([hmc.selector])
    end
end
