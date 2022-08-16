using CUDA
using FFTW
using Plots
using Flux
using CSV
using DataFrames
using Random
using Statistics


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

                ### PRINT INFO ABOUT GRADIENTS
                gs_list = [gs[p] for p in ps]
                for g in gs_list
                    println(typeof(g))
                    # g_vec = vec(g)
                    # std = Statistics.std(g_vec)
                    # mean = Statistics.mean(g_vec)
                    # println("std = ", std)
                    # println("mean = ", mean)
                end


            else
                (XR, Y, YR, YΘ, timestep_list), (XΘ,) = batch
	            XR, XΘ, YR, YΘ = device(XR), device(XΘ), device(YR), device(YΘ) 
                ŶR, ŶΘ = model(XR), model(XΘ)

                ### COMPUTATION OF LOSS IN ORIGINAL (NON-FOURIER) GRID-POINT CARTESIAN SPACE [TOO SLOW]
                #
                # Ŷ = VectorCartesianFromPolar(ŶR, ŶΘ)        # convert to Cartesian coors in Fourier space
                # Ŷ = reshape(Ŷ, (62, 62, args.batchsize))    # reshape to square grid
                # Ŷ = ifft(Ŷ, (1,2))                          # inverse fast fourier transform to compare against non-Fourier, Cartesian y
                # Ŷ = reshape(Ŷ, (3844, args.batchsize))      # reshape back into vector
                # gs = gradient(() -> mse(Ŷ, Y), ps)          ## compute gradient
                #

                # println("size(timestep)", size(timestep))
                # println("size(ŶR): ", size(ŶR))
                # println("size(YR): ", size(YR))
                # println("size(ŶΘ): ", size(ŶΘ))
                # println("size(YΘ): ", size(YΘ))

                # noise = device(rand(Float64, size(ŶR)))
                # gs = Flux.gradient(() -> mse(ŶR, ŶR + noise), ps)# + mse(ŶΘ, YΘ), ps) ## compute gradient

                # gs = Flux.gradient(() -> device(rand(Float64)), ps)# + mse(ŶΘ, YΘ), ps) ## compute gradient

                gs = Flux.gradient(() -> mse(ŶR, YR), ps)# + mse(ŶΘ, YΘ), ps) ## compute gradient
                
                ### SERIALIZE FLATTENED PARAMS TO CSV FILE TO SEE IF THEY'RE BEING UPDATED              
                # println(gs.grads)
                # for (i, p) in enumerate(ps)
                #     df = DataFrame(vec(Array(p))', :auto)  # flatten params to vector and transpose to row
                #     CSV.write("/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/csvs/ps_$i.csv", df, delim = ',',append=true)
                # end

                ### PRINT INFO ABOUT GRADIENTS
                gs_list = [gs[p] for p in ps]
                for g in gs_list
                    println(typeof(g))
                    # g_vec = vec(g)
                    # std = Statistics.std(g_vec)
                    # mean = Statistics.mean(g_vec)
                    # println("std = ", std)
                    # println("mean = ", mean)
                end
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
