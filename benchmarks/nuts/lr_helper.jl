using DelimitedFiles

function readlrdata()

    fname = joinpath(dirname(@__FILE__), "lr_nuts.data")
    z = readdlm(fname)
    x = z[:,1:end-1]
    y = z[:,end] .- 1
    return x, y
end
