using JLD2, FileIO
using FFTW
using Flux.Data: DataLoader


function getdata(args)

    data = load(args.data)

    if args.fourier
        RX_list, Ry_list = [], []
        ΘX_list, Θy_list = [], []
    else
        X_list = []
        y_list = []
    end
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
    for (timestep, timestep_data) in data

        if args.fourier

            ## fourier input
            if !args.cnn_input
                RX_instance, ΘX_instance =  VectorPolarFromCartesian(fft(timestep_data[timestep]["rhs"]))  # shape 3600
            else
                RX_instance, ΘX_instance =  VectorPolarFromCartesian(fft(reshape(timestep_data[timestep]["rhs"], (60, 60, 1))))  # shape (60, 60, 1)
            end
            push!(RX_list, RX_instance)
            push!(ΘX_list, ΘX_instance)

            ## fourier output
            Ry_instance, Θy_instance = VectorPolarFromCartesian(fft(reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,))))  # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
            push!(Ry_list, Ry_instance)
            push!(Θy_list, Θy_instance)

        else

            ## non-fourier input
            if !args.cnn_input
                push!(X_list, timestep_data[timestep]["rhs"])  # shape 3600
            else
                push!(X_list, reshape(timestep_data[timestep]["rhs"], (60, 60, 1)))  # shape (60, 60, 1)
            end

            ## non-fourier output
            push!(y_list, reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,)))    # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten

        end
    end

    ## Concatenate all samples
    if args.fourier
        RX = args.cnn_input ? cat(RX_list...; dims=4) : cat(RX_list...; dims=2)
        Ry = cat(Ry_list...; dims=2)
        ΘX = args.cnn_input ? cat(ΘX_list...; dims=4) : cat(ΘX_list...; dims=2)
        Θy = cat(Θy_list...; dims=2)
    else
        X = args.cnn_input ? cat(X_list...; dims=4) : cat(X_list...; dims=2)
        y = cat(y_list...; dims=2)
    end

    ## Create DataLoader object(s) (mini-batch iterator)
    if args.fourier
        R_loader = DataLoader((data=RX, label=Ry), batchsize=args.batchsize, shuffle=!args.decode)
	Θ_loader = DataLoader((data=ΘX, label=Θy), batchsize=args.batchsize, shuffle=!args.decode)
        return R_loader, Θ_loader
    else
        loader = DataLoader((data=X, label=y), batchsize=args.batchsize, shuffle=!args.decode)
        return loader
    end
    
end
