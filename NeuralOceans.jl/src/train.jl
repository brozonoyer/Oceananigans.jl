using CUDA
using FFTW
using Plots

function train(; kws...)

    args = Args(; kws...) ## Collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Create train dataloader
    if args.fourier
        train_R_loader, train_Θ_loader = getdata(args)
	    train_loader = zip(train_R_loader, train_Θ_loader)
    else
        train_loader = getdata(args)
    end

    ## Construct model
    model = load_model_from_json(args.model_config) |> device
    ps = Flux.params(model) ## model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)
    
    epoch_list, loss_list = [], []

    ## Training
    for epoch in 1:args.epochs
        for batch in train_loader
	        if !args.fourier
                (x, y, timestep) = batch
	            x, y = device(x), device(y)
                gs = gradient(() -> mse(model(x), y), ps)   ## compute gradient
            else
                ((Rx, y, Ry_fourier, Θy_fourier, timestep), (Θx,)) = batch
	            Rx, Θx, Ry_fourier, Θy_fourier = device(Rx), device(Θx), device(Ry_fourier), device(Θy_fourier)
                Rŷ, Θŷ = model(Rx), model(Θx)

                ### COMPUTATION OF LOSS IN ORIGINAL (NON-FOURIER) GRID-POINT CARTESIAN SPACE
                #
                # ŷ = VectorCartesianFromPolar(Rŷ, Θŷ)        # convert to Cartesian coors in Fourier space
                # ŷ = reshape(ŷ, (62, 62, args.batchsize))    # reshape to square grid
                # ŷ = ifft(ŷ, (1,2))                          # inverse fast fourier transform to compare against non-Fourier, Cartesian y
                # ŷ = reshape(ŷ, (3844, args.batchsize))      # reshape back into vector
                # gs = gradient(() -> mse(ŷ, y), ps)          ## compute gradient
                #

                # println("size(timestep)", size(timestep))
                # println("size(Rŷ): ", size(Rŷ))
                # println("size(Ry_fourier): ", size(Ry_fourier))
                # println("size(Θŷ): ", size(Θŷ))
                # println("size(Θy_fourier): ", size(Θy_fourier))

                gs = gradient(() -> mse(Rŷ, Ry_fourier) + mse(Θŷ, Θy_fourier), ps) ## compute gradient
            end
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        ## Report on train and test
        ## TODO compute test loss for display in original space (non-Fourier Cartesian)
        train_loss = loss(train_loader, model, device, args; fourier=args.fourier)
        println("Epoch=$epoch")
        push!(epoch_list, epoch)
        println("  train_loss = $train_loss")
        push!(loss_list, train_loss)
    end

    fourier_str = args.fourier ? "fourier-" : ""
    model_str = args.cnn_input ? "cnn" : "mlp"
    # p = plot(epoch_list[50:end], loss_list[50:end], title = "loss per epoch ($fourier_str$model_str)", label = ["epoch", "mse"], lw = 3)
    # savefig(p, "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/plots/$fourier_str$model_str-mse.png")

    return cpu(model)
end
