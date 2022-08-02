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

    # model = MLP() |> device
    model = load_model_from_json(args.model_config) |> device
    weights = BSON.load(model_path, @__MODULE__)
    model = Flux.loadmodel!(model, weights[:model])

    ## Create test dataloader
    if args.fourier
        test_R_loader, test_Θ_loader = getdata(args)
        test_loader = zip(test_R_loader, test_Θ_loader)
    else
        test_loader = getdata(args)
    end

    η_per_timestep = Dict()
    for batch in test_loader
        if args.fourier
            ((Rx, Ry), (Θx, Θy)) = batch
            Rx, Θx = device(Rx), device(Θx)
            Rŷ, Θŷ = model(Rx), model(Θx)
            η_per_timestep[timestep] = ifft(VectorCartesianFromPolar(Rŷ, Θŷ))
        else
            (x, y, timestep) = batch
            ŷ = model(device(x))
            η_per_timestep[timestep] = ŷ
        end
    end

    # save η_per_timestep to JLD2 file
    println("Saving to ", "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/predictions/η_per_timestep.jld2")
    @save "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/predictions/η_per_timestep.jld2" η_per_timestep

    return η_per_timestep

end
