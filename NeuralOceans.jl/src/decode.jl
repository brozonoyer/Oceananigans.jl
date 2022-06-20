using Flux
using BSON
using CUDA
using FFTW


function decode(model_path; kws...)
    
    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Decoding on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Decoding on CPU"
        device = cpu
    end

    model = MLP() |> device
    weights = BSON.load(model_path)
    model = Flux.loadmodel!(model, weights[:model])

    ## Create test dataloader
    test_loader = getdata(args)

    Ŷ = []
    for (x, y) in test_loader
        if args.fourier
            Rx, Θx = device(x[1][1]), device(x[1][2])
            Rŷ, Θŷ = model(Rx), model(Θx)
            push!(Ŷ, ifft(VectorCartesianFromPolar(Rŷ, Θŷ)))
        else
	    ŷ = model(device(x[1]))
            push!(Ŷ, ŷ)
        end
    end

    return Ŷ
end
