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
            ((Rx, y), (Θx, _)) = batch
            Rx, Θx = device(Rx), device(Θx)
            Rŷ, Θŷ = model(Rx), model(Θx)
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
