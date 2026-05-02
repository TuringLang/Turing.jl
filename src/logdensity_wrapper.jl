"""
    SafeLogDensity(ldf)

Return '-Inf' if evaluation throws an error.
"""

struct SafeLogDensity{L}
    ldf::L
end

function LogDensityProblems.logdensity(w::SafeLogDensity, x)
    try
        return LogDensityProblems.logdensity(w.ldf, x)
    catch
        return -Inf
    end
end