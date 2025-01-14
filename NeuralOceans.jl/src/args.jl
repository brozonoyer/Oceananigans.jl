using Base: @kwdef


@kwdef mutable struct Args
    model_config::String = "/nfs/nimble/users/brozonoy/Oceananigans.jl/NeuralOceans/jl/src/configs/mlp.config"  ## path to json model config
    data::String = "/nfs/nimble/users/brozonoy/Oceananigans.jl/tdata.jld2"      ## path to jld2 data file
    # η_per_timestep::String = "/nfs/nimble/users/brozonoy/NeuralOceans.jl/Oceananigans.jl/predictions"    ## path to output predictions per timestep during decoding
    η::Float64 = 3e-4           ## learning rate
    batchsize::Int = 6          ## batch size
    epochs::Int = 500           ## number of epochs
    use_cuda::Bool = true       ## use gpu (if cuda available)
    fourier::Bool = false       ## train model in fourier space
    cnn_input::Bool = false     ## if training CNN, input data can be 2d, otherwise it is flattened
    decode::Bool = false        ## do inference rather than training
end
