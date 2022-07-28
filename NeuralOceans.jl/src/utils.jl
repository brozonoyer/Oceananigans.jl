using CoordinateTransformations
using ZippedArrays: ZippedArray
using CUDA
using Statistics


function VectorPolarFromCartesian(Vcartesian)
    Vpolar = map(tup->PolarFromCartesian()([tup[1], tup[2]]), ZippedArray(real.(Vcartesian), imag.(Vcartesian)))
    R, Θ = map(n->n.r, Vpolar), map(n->n.θ, Vpolar)
    return R, Θ
end


function VectorCartesianFromPolar(VR, VΘ)
    CUDA.allowscalar() do
        return map(tup->tup[1]+tup[2]im, CartesianFromPolar().(map(tup->Polar(tup[1], tup[2]), ZippedArray(VR, VΘ))))
    end
end


function test_vectorized_coordinate_conversions()
    
    # create matrix of complex unit vectors
    C = [1+0im          0+1im ;
         1/√2+1/√2im    -√3/2-1im/2]

    # convert to polar
    R, Θ = VectorPolarFromCartesian(C)

    # magnitudes should all be 1
    @assert Statistics.mean(R) - 1 < 0.000001

    # phases should equal Θ_gold
    Θ_gold = [0     π ;
              π/4  -5π/6]
    @assert sum(Θ - Θ_gold) < 0.000001

    # convert back to cartesian and make sure we're back at the starting point
    Ĉ = VectorCartesianFromPolar(R, Θ)
    δ = sum(C - Ĉ)
    @assert √(real(δ)^2 + imag(δ)^2) < 0.000001

end


if abspath(PROGRAM_FILE) == @__FILE__
    test_vectorized_coordinate_conversions()
end
