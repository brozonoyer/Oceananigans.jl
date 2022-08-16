using Flux
using BSON
using CUDA
using FFTW
using JLD2


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
            ((XR, YR), (XΘ, YΘ)) = batch
            XR, ΘX = device(XR), device(ΘX)
            ŶR, ŶΘ = model(XR), model(XΘ)
            # η_per_timestep[timestep] = ifft(VectorCartesianFromPolar(ŶR, ŶΘ))
        else
            (X, Y, timestep_batch) = batch
            Ŷ = Array(model(device(X)))
            # println(size(Ŷ))
            for idx in 1:1:length(timestep_batch)
                η_per_timestep[timestep_batch[idx]] = Ŷ[:, idx]
            end
        end
    end

    fourier_str = args.fourier ? "fourier-" : ""
    model_str = args.cnn_input ? "cnn" : "mlp"

    # save η_per_timestep to JLD2 file
    println("Saving to ", "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/predictions/η_per_timestep.$fourier_str$model_str.jld2")
    JLD2.save("/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/predictions/η_per_timestep.$fourier_str$model_str.jld2", "η_per_timestep", η_per_timestep)
    # load with: JLD2.load("<PATH>")["η_per_timestep"]

    return η_per_timestep

end
