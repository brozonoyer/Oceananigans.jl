using Flux


function MLP(; inputsize=(3600,1))
    return Chain( Dense(prod(inputsize), 256, relu),
                Dense(256, 62*62))
end
