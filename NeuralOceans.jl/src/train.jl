using CUDA
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
                (x, y) = batch
	            x, y = device(x), device(y) ## transfer data to device
                # gs = gradient(() -> logitcrossentropy(model(x), y), ps) ## compute gradient
                gs = gradient(() -> mse(model(x), y), ps) ## compute gradient
            else
                ((Rx, Ry), (Θx, Θy)) = batch
	            Rx, Ry, Θx, Θy = device(Rx), device(Ry), device(Θx), device(Θy)
                # gs = gradient(() -> logitcrossentropy(model(Rx), Ry) + logitcrossentropy(model(Θx), Θy), ps) ## compute gradient
                gs = gradient(() -> mse(model(Rx), Ry) + mse(model(Θx), Θy), ps) ## compute gradient
            end
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        ## Report on train and test
        train_loss = loss(train_loader, model, device; fourier=args.fourier)
        println("Epoch=$epoch")
        push!(epoch_list, epoch)
        println("  train_loss = $train_loss")
        push!(loss_list, train_loss)
    end

    fourier_str = args.fourier ? "fourier-" : ""
    model_str = args.cnn_input ? "cnn" : "mlp"
    p = plot(epoch_list[50:end], loss_list[50:end], title = "loss per epoch ($fourier_str$model_str)", label = ["epoch", "mse"], lw = 3)
    savefig(p, "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/plots/$fourier_str$model_str-mse.png")

    return cpu(model)
end
