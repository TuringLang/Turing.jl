module RandomMeasures

using ..Core, ..Core.VarReplay, ..Utilities
using Distributions
using LinearAlgebra
using StatsFuns: logsumexp

import Distributions: sample, logpdf
import Base: maximum, minimum, rand

# include concrete implementations
include("rpm.jl")

end # end module
