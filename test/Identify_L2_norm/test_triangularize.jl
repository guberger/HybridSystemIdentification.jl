module TestMain

using Test
using LinearAlgebra
using StaticArrays
@static if isdefined(Main, :TestLocal)
    include("../../src/HybridSystemIdentification.jl")
else
    using HybridSystemIdentification
end
HSI = HybridSystemIdentification

sleep(0.1) # used for good printing
println("Started test")

@testset "Triangularize!" begin
    for m = 1:20
        for n = m:2*m
            A = [3*i - j + cos(i*j/sqrt(2)) for i = 1:m, j = 1:n]
            AH = Matrix(UpperHessenberg(A))
            R1 = qr(AH).R
            R2 = HSI._triangularize_hessenberg!(A)
            D1 = [R1[i, i] for i = 1:m]
            D2 = [R2[i, i] for i = 1:m]
            @test norm(abs.(D1) - abs.(D2)) < 1e-9*m
            S = D2 ./ D1
            @test norm(abs.(S) .- 1) < 1e-9*m
            R3 = S .\ R1
            for i = 2:m
                for j = 1:i-1
                    R2[i, j] = 0
                end
            end
            @test norm(R2 - R3) < 1e-9*m*n
        end
    end
end

sleep(0.1) # used for good printing
println("End test")

end  # module TestMain