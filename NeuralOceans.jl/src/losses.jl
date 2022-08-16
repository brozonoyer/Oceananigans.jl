using Flux.Losses: logitcrossentropy, mse


function loss(data_loader, model, device, args; fourier=false)
    ls = 0.0f0
    num = 0
    for batch in data_loader
        if !fourier
	        (X, Y) = batch
            X, Y = device(X), device(Y)
            Ŷ = model(X)
            ls += mse(Ŷ, Y, agg=sum)
            num += size(X)[end]
        else
            (XR, Y, YR, YΘ, timestep_list), (XΘ,) = batch
            XR, XΘ, YR, YΘ = device(XR), device(XΘ), device(YR), device(YΘ)
            ŶR, ŶΘ = model(XR), model(XΘ)

            ### COMPUTATION OF LOSS IN FOURIER POLAR SPACE
            # ls += mse(ŶR, YR) + mse(ŶΘ, YΘ)

            ### COMPUTATION OF LOSS IN ORIGINAL (NON-FOURIER) GRID-POINT CARTESIAN SPACE
            
            Ŷ = VectorCartesianFromPolar(ŶR, ŶΘ)        # convert to Cartesian coors in Fourier space
            Ŷ = reshape(Ŷ, (62, 62, args.batchsize))    # reshape to square grid
            Ŷ = ifft(Ŷ, (1,2))                          # inverse fast fourier transform to compare against non-Fourier, Cartesian y
            Ŷ = reshape(Ŷ, (3844, args.batchsize))      # reshape back into vector
            
            ls += mse(Ŷ, Y, agg=sum)
            
            num += size(XR)[end]

        end
    end
    return ls / num
end
