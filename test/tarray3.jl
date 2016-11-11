using Turing: TArray, localcopy, tzeros
using Base.Test

ta1 = TArray{Int}(4);
push!(ta1, 1);
push!(ta1, 2);
@test pop!(ta1) == 2

ta2 = TArray{Int}(4, 4);
ta3 = TArray{Int, 4}(4, 3, 2, 1);
ta4 = localcopy(ta3);

@test ta3[3] == ta4[3]

ta5 = TArray{Int}(4);
for i in 1:4 ta5[i] = i end
@test Array(ta5) == [1, 2, 3, 4]

@test Array(tzeros(4)) == zeros(4)
