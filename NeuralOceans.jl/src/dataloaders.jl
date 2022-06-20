using JLD2, FileIO
using FFTW
using Flux.Data: DataLoader


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
