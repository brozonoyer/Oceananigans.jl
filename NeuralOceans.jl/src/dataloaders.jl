using JLD2, FileIO
using FFTW
using Flux.Data: DataLoader


function getdata(args)

    data = load(args.data)
    """
    data organized by timestep:
    t1 => (
        t1 => (
            rhs => ...  # 3600
            η => ...    # 66, 66, 1
        )
    ),
    t2 => (...),
    ...
    """

    timestep_list, X, Y = [], [], []   # timesteps, RHS, \Eta

    ### FIRST PASS: COLLECT RAW DATA PER TIMESTEP
    for (timestep, timestep_data) in data

        push!(timestep_list, timestep)                                             # timestep: ℤ
        push!(X, reshape(timestep_data[timestep]["rhs"], (60, 60, 1)))             # rhs:      3600 -> (60, 60, 1)
        push!(Y, timestep_data[timestep]["η"][3:64, 3:64])                         # η:        (62, 62)

    end

    ### DO FOURIER AND CARTESIAN->POLAR CONVERSIONS
    if args.fourier

        ### SECOND PASS: APPLY FFT
        X_Fourier = map(rhs->fft(rhs, (1,2)), X)
        Y_Fourier = map(η->fft(η, (1,2)), Y)

        ### THIRD PASS: FOURIER CARTESIAN TO FOURIER POLAR
        X_Polar = map(rhs->VectorPolarFromCartesian(rhs), X_Fourier)      # [(XR,XΘ),...]
        Y_Polar = map(η->VectorPolarFromCartesian(η), Y_Fourier)          # [(YR,YΘ),...]

        XR = map(tup->tup[1], X_Polar)    # [XR,...]
        XΘ = map(tup->tup[2], X_Polar)    # [XΘ,...]

        YR = map(tup->tup[1], Y_Polar)    # [YR,...]
        YΘ = map(tup->tup[2], Y_Polar)    # [YΘ,...]

    end

    ### IF TRAINING MLP, EACH INSTANCE OF INPUT X SHOULD BE FLATTENED FROM (60,60,1) TO (3600,)
    if !args.cnn_input
        X = map(rhs->reshape(rhs, (60*60,)), X)
        if args.fourier
            XR = map(rhs->reshape(rhs, (60*60,)), XR)
            XΘ = map(rhs->reshape(rhs, (60*60,)), XΘ)
        end
    end
    ### EACH INSTANCE OF OUTPUT Y SHOULD BE FLATTENED FOR DENSE PREDICTION LAYER
    Y = map(η->reshape(η, (62*62,)), Y)             # cartesian non-fourier
    if args.fourier
        Y_Fourier = map(η->reshape(η, (62*62,)), Y_Fourier)     # cartesian fourier
        YR =        map(η->reshape(η, (62*62,)), YR)            # polar fourier magnitude
        YΘ =        map(η->reshape(η, (62*62,)), YΘ)            # polar fourier phase
    end

    ### CONCATENATE ALL SAMPLES
    ### MLP
    if !args.fourier
        X = args.cnn_input ? cat(X...; dims=4) : cat(X...; dims=2)
    ### CNN
    else
        ### POLAR DATA IS IN FOURIER SPACE
        XR = args.cnn_input ? cat(XR...; dims=4) : cat(XR...; dims=2)
        XΘ = args.cnn_input ? cat(XΘ...; dims=4) : cat(XΘ...; dims=2)
        YR = cat(YR...; dims=2)
        YΘ = cat(YΘ...; dims=2)
    end
    Y = cat(Y...; dims=2)

    fourier_str = args.fourier ? "fourier-" : ""
    model_str = args.cnn_input ? "cnn" : "mlp"

    ### CREATE DATALOADER OBJECT(S) (MINI-BATCH ITERATOR)
    if !args.fourier
        loader = DataLoader((data=X, Y=Y, timestep=timestep_list), batchsize=args.batchsize, shuffle=!args.decode)
        FileIO.save("/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/jld2/$fourier_str$model_str.jld2", Dict("X"=>X, "Y"=>Y,))
        return loader
    else
        R_loader = DataLoader((data=XR, Y=Y, YR=YR, YΘ=YΘ, timestep=timestep_list), batchsize=args.batchsize, shuffle=!args.decode)
	    Θ_loader = DataLoader((data=XΘ,), batchsize=args.batchsize, shuffle=!args.decode)
        FileIO.save("/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans.jl/jld2/$fourier_str$model_str.jld2", Dict("XR"=>XR, "XΘ"=>XΘ, "Y"=>Y, "YR"=>YR, "YΘ"=>YΘ))
        return R_loader, Θ_loader
    end
    
end
