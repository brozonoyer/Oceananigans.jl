using CoordinateTransformations
using ZippedArrays: ZippedArray


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
