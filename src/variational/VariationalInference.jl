module Variational

using DynamicPPL: DynamicPPL
using StatsBase: StatsBase
using LogDensityProblems: LogDensityProblems
using Distributions
using DistributionsAD
using Optimisers: Adam
using StatsFuns: StatsFuns

using Random: Random

import AdvancedVI
import Bijectors


# Reexports
using AdvancedVI: optimize, ADVI
export vi, ADVI

# VI algorithms
include("advi.jl")

end
