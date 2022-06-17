using JLD2, FileIO
using ArgParse
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using FFTW
using CoordinateTransformations
using ZippedArrays
using Base: @kwdef
using CUDA
using MLDatasets
using BSON
using BSON: @save, @load


@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 1    ## batch size
    epochs::Int = 50        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    data::String = "/nfs/nimble/users/brozonoy/Oceananigans.jl/tdata.jld2"      ## path to jld2 data file
    fourier::Bool = false  # train model in fourier space
    decode::Bool = false  # do inference rather than training
end


function getdata(args)

    data = load(args.data)

    X = []
    y = []
    for (timestep, timestep_data) in data
        if args.fourier
            RX, ΘX =  VectorPolarFromCartesian(fft(timestep_data[timestep]["rhs"]))  # shape 3600
            push!(X, (RX, ΘX))
            Ry, Θy = VectorPolarFromCartesian(fft(reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,))))  # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
            push!(y, (Ry, Θy))
        else
            push!(X, timestep_data[timestep]["rhs"])  # shape 3600
            push!(y, reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,)))    # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
        end
    end
    
    ## Create DataLoader object (mini-batch iterator)
    if args.decode
        loader = DataLoader((X, y), batchsize=args.batchsize, shuffle=false)
    else
        loader = DataLoader((X, y), batchsize=args.batchsize, shuffle=true)
    end
    return loader
end


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


function MLP(; inputsize=(3600,1))
    return Chain( Dense(prod(inputsize), 256, relu),
                Dense(256, 62*62))
end


function loss(data_loader, model, device; fourier=false)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        if !fourier
            x, y = device(x[1]), device(y[1])
            ŷ = model(x)
            ls += logitcrossentropy(ŷ, y, agg=sum)
            num +=  size(x)[end]
        else
            Rx, Θx, Ry, Θy = device(x[1][1]), device(x[1][2]), device(y[1][1]), device(y[1][2])
            Rŷ, Θŷ = model(Rx), model(Θx)
            ls += (logitcrossentropy(Rŷ, Ry, agg=sum) + logitcrossentropy(Θŷ, Θy, agg=sum))
            num +=  2*size(Rx)[end]
        end
    end
    return ls / num
end


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
    model = MLP() |> device
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
    #println(model_path) 
    #println(@load(model_path))
    weights = BSON.load(model_path)
    println(weights)
    model = Flux.loadmodel!(model, weights[:model])

    ## Create test dataloader
    test_loader = getdata(args)

    Ŷ = []
    for (x, y) in test_loader
        if args.fourier
            Rx, Θx = device(x[1][1]), device(x[1][2])
            Rŷ, Θŷ = model(Rx), model(Θx)
	    println(typeof(VectorCartesianFromPolar(Rŷ, Θŷ)))
	    println(size(VectorCartesianFromPolar(Rŷ, Θŷ)))
            push!(Ŷ, ifft(VectorCartesianFromPolar(Rŷ, Θŷ)))
        else
            push!(Ŷ, ŷ)
        end
    end

    return Ŷ
end


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--data", "-i"
            help = "path to jld2 in jld2 format"
            arg_type = String
            required = true
        "--model", "-o"
            help = "path to save model"
            arg_type = String
            required = true
        "--decode", "-d"
            help = "do inference rather than training"
            action = :store_true
        "--fourier", "-f"
            help = "whether model trains/decodes in fourier space"
            action = :store_true
    end

    return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__

    parsed_args = parse_commandline()
    if parsed_args["decode"]
        decode(parsed_args["model"];  data=parsed_args["data"], fourier=parsed_args["fourier"])
    else
        model = train(; data=parsed_args["data"], fourier=parsed_args["fourier"])
        @save parsed_args["model"] model
    end

end
