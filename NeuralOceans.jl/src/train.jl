using CUDA


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
    train_loader = getdata(args)

    ## Construct model
    # model = MLP() |> device
    model = load_model_from_json(args.model_config) |> device
    ps = Flux.params(model) ## model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)
    
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            if !args.fourier
                x, y = device(x[1]), device(y[1]) ## transfer data to device
                gs = gradient(() -> logitcrossentropy(model(x), y), ps) ## compute gradient
            else
                Rx, Θx, Ry, Θy = device(x[1][1]), device(x[1][2]), device(y[1][1]), device(y[1][2])
                gs = gradient(() -> logitcrossentropy(model(Rx), Ry) + logitcrossentropy(model(Θx), Θy), ps) ## compute gradient
            end
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        ## Report on train and test
        train_loss = loss(train_loader, model, device; fourier=args.fourier)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss")
    end

    return cpu(model)
end
