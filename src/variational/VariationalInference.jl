module Variational

using DynamicPPL: DynamicPPL
using StatsBase: StatsBase
using LogDensityProblems: LogDensityProblems
using Distributions
using DistributionsAD
using Optimisers: Adam
using StatsFuns: StatsFuns
import ..Essential: ADBackend

using Random: Random

import AdvancedVI
import Bijectors

using AdvancedVI: optimize, ADVI

function vi end

export vi, ADVI

# VI algorithms
include("advi.jl")

end
