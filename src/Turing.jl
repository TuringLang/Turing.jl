module Turing

using Reexport, ForwardDiff
using Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
using Distributions
@reexport using MCMCChains
using Compat: pkgversion

using AdvancedVI: AdvancedVI
using DynamicPPL: DynamicPPL
import DynamicPPL: NoDist, NamedDist
using LogDensityProblems: LogDensityProblems
using StatsAPI: StatsAPI
using StatsBase: StatsBase
using AbstractMCMC

using Printf: Printf
using Random: Random
using LinearAlgebra: I

using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

const DEFAULT_ADTYPE = ADTypes.AutoForwardDiff()

const PROGRESS = Ref(true)

# TODO: remove `PROGRESS` and this function in favour of `AbstractMCMC.PROGRESS`
"""
    setprogress!(progress::Bool)

Enable progress logging in Turing if `progress` is `true`, and disable it otherwise.
"""
function setprogress!(progress::Bool)
    @info "[Turing]: progress logging is $(progress ? "enabled" : "disabled") globally"
    PROGRESS[] = progress
    AbstractMCMC.setprogress!(progress; silent=true)
    return progress
end

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")
include("common.jl")
include("mcmc/Inference.jl")  # inference algorithms
using .Inference
include("variational/Variational.jl")
using .Variational

include("optimisation/Optimisation.jl")
using .Optimisation

###########
# Exports #
###########
# `using` statements for stuff to re-export
using DynamicPPL:
    @model,
    @varname,
    pointwise_logdensities,
    pointwise_loglikelihoods,
    pointwise_prior_logdensities,
    generated_quantities,
    returned,
    logprior,
    logjoint,
    condition,
    decondition,
    fix,
    unfix,
    prefix,
    conditioned,
    to_submodel,
    LogDensityFunction,
    VarNamedTuple,
    @vnt,
    @addlogprob!,
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    setthreadsafe,
    filldist,
    arraydist,
    set_logprob_type!

using StatsBase: predict
using OrderedCollections: OrderedDict
using Libtask: might_produce, @might_produce

# Selective re-export from Distributions
# We export all distribution types and essential functions,
# but avoid exporting internal helpers unlikely to be used by Turing users.
export
    # Generic distribution types
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
    DiscreteUnivariateDistribution,
    ContinuousUnivariateDistribution,
    DiscreteMultivariateDistribution,
    ContinuousMultivariateDistribution,
    DiscreteMatrixDistribution,
    ContinuousMatrixDistribution,
    Univariate,
    Multivariate,
    Matrixvariate,
    Discrete,
    Continuous,
    Sampleable,
    VariateForm,
    ValueSupport,
    ArrayLikeVariate,
    CholeskyVariate,
    NamedTupleVariate,
    NonMatrixDistribution,
    AbstractMvNormal,
    AbstractMixtureModel,
    UnivariateMixture,
    MultivariateMixture,
    # Distribution types
    Arcsine,
    Bernoulli,
    BernoulliLogit,
    Beta,
    BetaBinomial,
    BetaPrime,
    Binomial,
    Biweight,
    Categorical,
    Cauchy,
    Chernoff,
    Chi,
    Chisq,
    Cosine,
    DiagNormal,
    DiagNormalCanon,
    Dirac,
    Dirichlet,
    DirichletMultinomial,
    DiscreteUniform,
    DiscreteNonParametric,
    DoubleExponential,
    Erlang,
    Epanechnikov,
    Exponential,
    FDist,
    FisherNoncentralHypergeometric,
    Frechet,
    FullNormal,
    FullNormalCanon,
    Gamma,
    GeneralizedPareto,
    GeneralizedExtremeValue,
    Geometric,
    Gumbel,
    Hypergeometric,
    InverseWishart,
    InverseGamma,
    InverseGaussian,
    IsoNormal,
    IsoNormalCanon,
    JohnsonSU,
    JointOrderStatistics,
    Kolmogorov,
    KSDist,
    KSOneSided,
    Kumaraswamy,
    Laplace,
    Levy,
    Lindley,
    LKJ,
    LKJCholesky,
    LocationScale,
    Logistic,
    LogNormal,
    LogUniform,
    LogitNormal,
    MvLogitNormal,
    MatrixBeta,
    MatrixFDist,
    MatrixNormal,
    MatrixTDist,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
    MvLogNormal,
    MvNormal,
    MvNormalCanon,
    MvNormalKnownCov,
    MvTDist,
    NegativeBinomial,
    NoncentralBeta,
    NoncentralChisq,
    NoncentralF,
    NoncentralHypergeometric,
    NoncentralT,
    Normal,
    NormalCanon,
    NormalInverseGaussian,
    OrderStatistic,
    Pareto,
    PGeneralizedGaussian,
    SkewedExponentialPower,
    Poisson,
    PoissonBinomial,
    Rayleigh,
    Rician,
    Semicircle,
    Skellam,
    SkewNormal,
    Soliton,
    StudentizedRange,
    SymTriangularDist,
    TDist,
    TriangularDist,
    Triweight,
    Truncated,
    Uniform,
    UnivariateGMM,
    VonMises,
    VonMisesFisher,
    WalleniusNoncentralHypergeometric,
    Weibull,
    Wishart,
    ZeroMeanIsoNormal,
    ZeroMeanIsoNormalCanon,
    ZeroMeanDiagNormal,
    ZeroMeanDiagNormalCanon,
    ZeroMeanFullNormal,
    ZeroMeanFullNormalCanon,
    # Essential functions for Turing users
    logpdf,
    logpdf!,
    pdf,
    cdf,
    ccdf,
    logcdf,
    logccdf,
    logdiffcdf,
    quantile,
    cquantile,
    invlogcdf,
    invlogccdf,
    truncated,
    censored,
    product_distribution,
    insupport,
    support,
    sampler,
    fit,
    fit_mle,
    params,
    partype,
    mean,
    median,
    var,
    std,
    skewness,
    kurtosis,
    entropy,
    mode,
    modes,
    moment,
    probs,
    succprob,
    failprob,
    rate,
    scale,
    shape,
    location,
    concentration,
    dof,
    span,
    component,
    components,
    componentwise_pdf,
    componentwise_logpdf,
    ncomponents,
    kldivergence,
    gradlogpdf,
    convolve,
    cf,
    mgf,
    cgf,
    probval