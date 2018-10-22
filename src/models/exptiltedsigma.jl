struct ExpTiltedSigma <: ContinuousUnivariateDistribution
    a::Float64
    l::Float64
end

function Distributions.rand(d::ExpTiltedSigma)
    a = d.a;
    l = d.l;
    
    g = (cos(a*pi/2))^(1/a)
    
    m = max(1, round(Int, l^a))
    S = zeros(Float64, m)

    for k in 1:m

        Sk = 0.0
        U = 2.0
        
        while U > exp(-l * Sk)
            Sk = stblrnd(a, 1., g/m^(1/a), floor(a)/m)
            U = rand()
        end

        S[k] = Sk
    end

    return sum(S)
end

function stblrnd(alpha::Float64, beta::Float64, gamma::Float64, delta::Float64)
    sizeOut = 1
   
    # TODO: Add tests to check if the implementations are correct.
    halfpi = pi/2.0

    if alpha == 2 # Gaussian distribution
        #r = sqrt(2) * randn(sizeOut)
        r = sqrt(2.0) * randn()
    elseif alpha == 1 && beta == 0  # Cauchy distribution
        #r = tan( pi/2 * (2*rand(sizeOut) - 1) )
        r = tan( halfpi * (2.0 * rand() - 1.0) ) 
    elseif alpha == .5 && abs(beta) == 1 # Levy distribution (a.k.a. Pearson V)
        #r = beta ./ randn(sizeOut).^2
        r = beta / randn()^2
    elseif beta == 0 # Symmetric alpha-stable
        #V = pi/2 * (2*rand(sizeOut) - 1)
        V = halfpi * (2.0*rand() - 1.0)
        #W = -log(rand(sizeOut))
        W = -log(rand())
        r = sin(alpha * V) / ( cos(V)^(1.0 / alpha) ) * ( cos( V*(1.0-alpha) ) / W )^( (1.0-alpha)/alpha )
    elseif alpha != 1 # General case, alpha not 1
        #V = pi/2 * (2*rand(sizeOut) - 1)
        V = halfpi * (2.0*rand() - 1.0)
        W = -log(rand())
        const_tmp = beta * tan(pi*alpha/2)
        B = atan(const_tmp)
        S = (1.0 + const_tmp * const_tmp)^(1.0/(2.0*alpha))
        r = S * sin( alpha*V + B ) / ( cos(V) )^(1.0/alpha) * ( cos( (1.0-alpha) * V - B ) / W )^((1.0-alpha)/alpha)
    else # General case, alpha = 1
        V = halfpi * (2*rand() - 1)
        W = -log(rand())
        sclshftV =  halfpi + beta * V
        r = 1/halfpi * ( sclshftV * tan(V) - beta * log( (halfpi * W * cos(V) ) / sclshftV ) )
    end

    # Scale and shift
    if alpha != 1
        r = gamma * r + delta
    else
        r = gamma * r + (2/pi) * beta * gamma * log(gamma) + delta
    end

    return r
end
