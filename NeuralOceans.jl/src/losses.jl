using Flux.Losses: logitcrossentropy


function loss(data_loader, model, device; fourier=false)
    ls = 0.0f0
    num = 0
    for batch in data_loader
        if !fourier
	    (x, y) = batch
            x, y = device(x), device(y)
            ŷ = model(x)
            ls += logitcrossentropy(ŷ, y, agg=sum)
            num +=  size(x)[end]
        else
            ((Rx, Ry), (Θx, Θy)) = batch
            Rx, Ry, Θx, Θy = device(Rx), device(Ry), device(Θx), device(Θy)
            Rŷ, Θŷ = model(Rx), model(Θx)
            ls += (logitcrossentropy(Rŷ, Ry, agg=sum) + logitcrossentropy(Θŷ, Θy, agg=sum))
            num +=  2*size(Rx)[end]
        end
    end
    return ls / num
end
