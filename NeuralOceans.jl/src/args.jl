using Base: @kwdef


@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 1    ## batch size
    epochs::Int = 50        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    data::String = "/nfs/nimble/users/brozonoy/Oceananigans.jl/tdata.jld2"      ## path to jld2 data file
    fourier::Bool = false  # train model in fourier space
    decode::Bool = false  # do inference rather than training
end
