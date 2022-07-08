using JLD2, FileIO
using FFTW
using Flux.Data: DataLoader


function getdata(args)

    data = load(args.data)

    X = []
    y = []
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
            RX, ΘX =  VectorPolarFromCartesian(fft(timestep_data[timestep]["rhs"]))  # shape 3600
            push!(X, (RX, ΘX))
            if !args.cnn_input
                Ry, Θy = VectorPolarFromCartesian(fft(reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,))))  # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
            else
                Ry, Θy = VectorPolarFromCartesian(fft(timestep_data[timestep]["η"]))  # shape (66, 66, 1) assume padding 2 on each side to ignore zeros in input
            end
            push!(y, (Ry, Θy))

        else
            push!(X, timestep_data[timestep]["rhs"])  # shape 3600
            if !args.cnn_input
                push!(y, reshape(timestep_data[timestep]["η"][3:64, 3:64], (62*62,)))    # shape (66, 66, 1) -> (62, 62, 1) -> 62*62=3844 strip away zeros and flatten
            else
                push!(y, timestep_data[timestep]["η"])    # shape (66, 66, 1) assume padding 2 on each side to ignore zeros in input
            end
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
