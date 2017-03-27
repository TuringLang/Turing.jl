# Code adapted from: https://github.com/johnmyleswhite/ASCIIPlots.jl/blob/master/src/ASCIIPlots.jl

immutable ASCIIPlot
    s::ASCIIString
end

Base.show(io::IO, p::ASCIIPlot) = print(io, p.s)

function scatterplot(x::AbstractArray, y::AbstractArray; sym::Char = '^')
    x, y = vec(x), vec(y)

    # Sanity checking
    N = length(x)
    if N != length(y)
        error("x and y must have the same length")
    end

    # Resolution along x and y dimensions
    res_x, res_y = 60, 20

    # Standarize data scale
    minx = minimum(x)
    maxx = maximum(x)
    miny = minimum(y)
    maxy = maximum(y)
    x = x .- minx
    x = x / maximum(x)
    y = y .- miny
    y = y / maximum(y)

    # Snap data points to a grid
    xi = floor(Integer, x * (res_x - 1)) .+ 1
    yi = floor(Integer, y * (res_y - 1)) .+ 1

    # Is there a point at location (i, j)?
    A = zeros(res_y, res_x)
    for i in 1:N
        A[yi[i], xi[i]] = 1
    end

    io = IOBuffer()

    print(io, "\n")

    # Top grid line
    print(io, "\t")
    for j = 1:(res_x + 1)
        print(io, "-")
    end
    print(io, "\n")

    for i = res_y:-1:1
        # Left grid line
        print(io, "\t|")

        # Data points
        for j = 1:res_x
            if A[i, j] == 1
                print(io, sym)
            else
                print(io, " ")
            end
        end

        # Right grid line + Y tick marks
        if i == res_y
            @printf io "| %2.2f\n" maxy
        elseif i == 1
            @printf io "| %2.2f\n" miny
        else
            print(io, "|\n")
        end
    end

    # Bottom grid line
    print(io, "\t")
    for j = 1:(res_x + 1)
        print(io, "-")
    end
    print(io, "\n")

    # Tick marks for X axis
    @printf io "\t%2.2f" minx
    for j = 1:(res_x - 8)
        print(io, " ")
    end
    @printf io "%2.2f" maxx

    print(io, "\n")

    return ASCIIPlot(bytestring(io))
end

function scatterplot(y::AbstractArray; sym::Char = '^')
    scatterplot([1:length(y)], y, sym = sym)
end
