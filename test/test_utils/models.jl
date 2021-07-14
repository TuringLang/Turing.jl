# The old-gdemo model.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

@model gdemo_d() = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

gdemo_default = gdemo_d()

@model MoGtest(D) = begin
    mu1 ~ Normal(1, 1)
    mu2 ~ Normal(4, 1)
    z1 ~ Categorical(2)
    if z1 == 1
        D[1] ~ Normal(mu1, 1)
    else
        D[1] ~ Normal(mu2, 1)
    end
    z2 ~ Categorical(2)
    if z2 == 1
        D[2] ~ Normal(mu1, 1)
    else
        D[2] ~ Normal(mu2, 1)
    end
    z3 ~ Categorical(2)
    if z3 == 1
        D[3] ~ Normal(mu1, 1)
    else
        D[3] ~ Normal(mu2, 1)
    end
    z4 ~ Categorical(2)
    if z4 == 1
        D[4] ~ Normal(mu1, 1)
    else
        D[4] ~ Normal(mu2, 1)
    end
    z1, z2, z3, z4, mu1, mu2
end

MoGtest_default = MoGtest([1.0 1.0 4.0 4.0])

# Declare empty model to make the Sampler constructor work.
@model empty_model() = begin x = 1; end

# A collection of models for which the mean-of-means for the posterior should
# be same.
@model function gdemo1(x = 10 * ones(2), ::Type{TV} = Vector{Float64}) where {TV}
    # `dot_assume` and `observe`
    m = TV(undef, length(x))
    m .~ Normal()
    x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo2(x = 10 * ones(2), ::Type{TV} = Vector{Float64}) where {TV}
    # `assume` with indexing and `observe`
    m = TV(undef, length(x))
    for i in eachindex(m)
        m[i] ~ Normal()
    end
    x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo3(x = 10 * ones(2))
    # Multivariate `assume` and `observe`
    m ~ MvNormal(length(x), 1.0)
    x ~ MvNormal(m, 0.5 * ones(length(x)))
end

@model function gdemo4(x = 10 * ones(2), ::Type{TV} = Vector{Float64}) where {TV}
    # `dot_assume` and `observe` with indexing
    m = TV(undef, length(x))
    m .~ Normal()
    for i in eachindex(x)
        x[i] ~ Normal(m[i], 0.5)
    end
end

# Using vector of `length` 1 here so the posterior of `m` is the same
# as the others.
@model function gdemo5(x = 10 * ones(1))
    # `assume` and `dot_observe`
    m ~ Normal()
    x .~ Normal(m, 0.5)
end

@model function gdemo6()
    # `assume` and literal `observe`
    m ~ MvNormal(2, 1.0)
    [10.0, 10.0] ~ MvNormal(m, 0.5 * ones(2))
end

@model function gdemo7(::Type{TV} = Vector{Float64}) where {TV}
    # `dot_assume` and literal `observe` with indexing
    m = TV(undef, 2)
    m .~ Normal()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end
end

@model function gdemo8()
    # `assume` and literal `dot_observe`
    m ~ Normal()
    [10.0, ] .~ Normal(m, 0.5)
end

@model function _prior_dot_assume(::Type{TV} = Vector{Float64}) where {TV}
    m = TV(undef, 2)
    m .~ Normal()

    return m
end

@model function gdemo9()
    # Submodel prior
    m = @submodel _prior_dot_assume()
    for i in eachindex(m)
        10.0 ~ Normal(m[i], 0.5)
    end
end

@model function _likelihood_dot_observe(m, x)
    x ~ MvNormal(m, 0.5 * ones(length(m)))
end

@model function gdemo10(x = 10 * ones(2), ::Type{TV} = Vector{Float64}) where {TV}
    m = TV(undef, length(x))
    m .~ Normal()

    # Submodel likelihood
    @submodel _likelihood_dot_observe(m, x)
end

const gdemo_models = (gdemo1(), gdemo2(), gdemo3(), gdemo4(), gdemo5(), gdemo6(), gdemo7(), gdemo8(), gdemo9(), gdemo10())
