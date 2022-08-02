using JLD2, FileIO
using FFTW
using Flux.Data: DataLoader


function getdata(args)

    data = load(args.data)

    timestep_list = []
    if args.fourier
        RX_list, ΘX_list = [], []
    else
        X_list = []
    end
    Ry_fourier_list, Θy_fourier_list, y_list = [], [], []  # training y in polar fourier, testing y in original cartesian space
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

        # record timestep
        push!(timestep_list, timestep)

        if args.fourier

            ## fourier input
            if !args.cnn_input
                rhs = timestep_data[timestep]["rhs"]  # right hand side for timestep, shape 3600
                rhs = reshape(fft(reshape(rhs, (60, 60)), (1,2)), 3600)  # perform 2d FFT and reshape back to original shape 3600
                RX_instance, ΘX_instance = VectorPolarFromCartesian(rhs)  # shape 3600
            else
                rhs = timestep_data[timestep]["rhs"]  # right hand side for timestep, shape 3600
                rhs = fft(reshape(rhs, (60, 60, 1)), (1,2))  # perform 2d FFT
                RX_instance, ΘX_instance = VectorPolarFromCartesian(rhs)  # shape (60, 60, 1)
            end
            push!(RX_list, RX_instance)
            push!(ΘX_list, ΘX_instance)

            ## fourier-space output (for training only, use cartesian-space y for testing)
            η_instance = reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,))
            η_fourier_instance = fft(η_instance, 1)
            Ry_fourier_instance, Θy_fourier_instance = VectorPolarFromCartesian(η_fourier_instance)  ## use just for training in fourier space
            push!(Ry_fourier_list, Ry_fourier_instance)
            push!(Θy_fourier_list, Θy_fourier_instance)

        else

            ## non-fourier input
            if !args.cnn_input
                push!(X_list, timestep_data[timestep]["rhs"])  # shape 3600
            else
                push!(X_list, reshape(timestep_data[timestep]["rhs"], (60, 60, 1)))  # shape (60, 60, 1)
            end

        end

        ## output
        push!(y_list, reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,)))    # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten

    end

    ## Concatenate all samples
    if args.fourier
        RX = args.cnn_input ? cat(RX_list...; dims=4) : cat(RX_list...; dims=2)
        Ry_fourier = cat(Ry_fourier_list...; dims=2)
        ΘX = args.cnn_input ? cat(ΘX_list...; dims=4) : cat(ΘX_list...; dims=2)
        Θy_fourier = cat(Θy_fourier_list...; dims=2)
    else
        X = args.cnn_input ? cat(X_list...; dims=4) : cat(X_list...; dims=2)
    end
    y = cat(y_list...; dims=2)

    ## Create DataLoader object(s) (mini-batch iterator)
    if args.fourier
        R_loader = DataLoader((data=RX, label=y, Ry_fourier=Ry_fourier, Θy_fourier=Θy_fourier, timestep=timestep_list), batchsize=args.batchsize, shuffle=!args.decode)
	    Θ_loader = DataLoader((data=ΘX,), batchsize=args.batchsize, shuffle=!args.decode)
        return R_loader, Θ_loader
    else
        loader = DataLoader((data=X, label=y, timestep=timestep_list), batchsize=args.batchsize, shuffle=!args.decode)
        return loader
    end
    
end
