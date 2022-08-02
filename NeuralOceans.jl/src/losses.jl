using Flux.Losses: logitcrossentropy, mse


function loss(data_loader, model, device, args; fourier=false)
    ls = 0.0f0
    num = 0
    for batch in data_loader
        if !fourier
	        (x, y) = batch
            x, y = device(x), device(y)
            ŷ = model(x)
            ls += mse(ŷ, y, agg=sum)
            num += size(x)[end]
        else
            ((Rx, y, Ry_fourier, Θy_fourier, timestep), (Θx,)) = batch
            Rx, Θx, Ry_fourier, Θy_fourier = device(Rx), device(Θx), device(Ry_fourier), device(Θy_fourier)
            Rŷ, Θŷ = model(Rx), model(Θx)

            ### COMPUTATION OF LOSS IN FOURIER POLAR SPACE
            # ls += mse(Rŷ, Ry_fourier) + mse(Θŷ, Θy_fourier)

            ### COMPUTATION OF LOSS IN ORIGINAL (NON-FOURIER) GRID-POINT CARTESIAN SPACE
            
            ŷ = VectorCartesianFromPolar(Rŷ, Θŷ)        # convert to Cartesian coors in Fourier space
            ŷ = reshape(ŷ, (62, 62, args.batchsize))    # reshape to square grid
            ŷ = ifft(ŷ, (1,2))                          # inverse fast fourier transform to compare against non-Fourier, Cartesian y
            ŷ = reshape(ŷ, (3844, args.batchsize))      # reshape back into vector
            
            ls += mse(ŷ, y, agg=sum)
            
            num += size(Rx)[end]

        end
    end
    return ls / num
end
