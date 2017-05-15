using Gadfly
Gadfly.push_theme(:dark)

N = 3
spls = ["HMC(1000,0.25,6)","HMCDA(1000,200,0.65,1.5)","NUTS(1000,200,0.65)"][1:N]
spls_un = ["Gibbs(1000,PG(50,1,:z),HMC(1,0.25,6,:phi,:theta))",
           "Gibbs(1000,PG(50,1,:z),HMCDA(1,200,0.65,1.5,:phi,:theta))",
           "Gibbs(1000,PG(50,1,:z),NUTS(1,200,0.65,:phi,:theta))"]

spl_colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d"][1:N]
