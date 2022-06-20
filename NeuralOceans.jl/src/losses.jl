using Flux.Losses: logitcrossentropy


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
