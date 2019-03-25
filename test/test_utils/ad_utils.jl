using Turing: gradient_logp_forward, gradient_logp_reverse
using Test

function test_ad(f, at = 0.5; rtol = 1e-8, atol = 1e-8)
    isarr = isa(at, AbstractArray)
    reverse = Tracker.gradient(f, at)[1]
    if isarr
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(reverse, forward, rtol=rtol, atol=atol)
    else
        forward = ForwardDiff.derivative(f, at)
        finite_diff = central_fdm(5,1)(f, at)
        @test isapprox(reverse, forward, rtol=rtol, atol=atol)
        @test isapprox(reverse, finite_diff, rtol=rtol, atol=atol)
    end
end

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo()
    model(vi, SampleFromPrior())

    # Collect symbols.
    vnms = Vector(undef, length(syms))
    vnvals = Vector{Float64}()
    for i in 1:length(syms)
        s = syms[i]
        vnms[i] = collect(
            Iterators.filter(vn -> vn.sym == s, keys(vi))
        )[1]

        vals = getval(vi, vnms[i])
        for i in eachindex(vals)
            push!(vnvals, vals[i])
        end
    end

    spl = SampleFromPrior()
    _, ∇E = gradient_logp_forward(vi[spl], vi, model)
    grad_Turing = sort(∇E)

    # Call ForwardDiff's AD
    grad_FWAD = sort(ForwardDiff.gradient(f, vec(vnvals)))

    # Compare result
    @test grad_Turing ≈ grad_FWAD atol=1e-9
end
