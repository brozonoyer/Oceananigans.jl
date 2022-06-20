module NeuralOceans

export train, decode, args

include("./args.jl")
include("./dataloaders.jl")
include("./decode.jl")
include("./losses.jl")
include("./models.jl")
include("./train.jl")
include("./utils.jl")

end