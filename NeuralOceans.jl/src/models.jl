using Flux, JSON
using NeuralOperators


# https://fluxml.ai/Flux.jl/stable/models/nnlib/
global activation_fns_lookup = Dict(
    "identity" => Flux.identity,
    "celu" => Flux.celu,
    "elu" => Flux.elu,
    "gelu" => Flux.gelu,
    "hardsigmoid" => Flux.hardsigmoid,
    "hardtanh" => Flux.hardtanh,
    "leakyrelu" => Flux.leakyrelu,
    "lisht" => Flux.lisht,
    "logcosh" => Flux.logcosh,
    "logsigmoid" => Flux.logsigmoid,
    "mish" => Flux.mish,
    "relu" => Flux.relu,
    "relu6" => Flux.relu6,
    "rrelu" => Flux.rrelu,
    "selu" => Flux.selu,
    "sigmoid" => Flux.sigmoid,
    "softplus" => Flux.softplus,
    "softshrink" => Flux.softshrink,
    "softsign" => Flux.softsign,
    "swish" => Flux.swish,
    "tanhshrink" => Flux.tanhshrink,
    "trelu" => Flux.trelu
)


# https://fluxml.ai/Flux.jl/stable/utilities/
global initialization_fns_lookup = Dict(
    "glorot_uniform" => Flux.glorot_uniform,
    "glorot_normal" => Flux.glorot_normal,
    "kaiming_uniform" => Flux.kaiming_uniform,
    "kaiming_normal" => Flux.kaiming_normal,
    "truncated_normal" => Flux.truncated_normal,
    "orthogonal" => Flux.orthogonal,
    "sparse_init" => Flux.sparse_init,
    "identity_init" => Flux.identity_init,
    "ones32" => Flux.ones32,
    "rand32" => Flux.rand32
)


function load_model_from_json(config_json_path::String)

    # json to Dict
    config = JSON.parse(read(open(config_json_path, "r"), String))

    # load every layer individually
    nn_layers = []
    for layer_config in config["layers"]
        if layer_config["layer"] == "Dense"
            nn_layer = dense_layer_from_config(layer_config)
        elseif layer_config["layer"] == "Conv"
            nn_layer = conv_layer_from_config(layer_config)
        elseif layer_config["layer"] == "MaxPool"
            nn_layer = max_pool_layer_from_config(layer_config)
        elseif layer_config["layer"] == "flatten"
            nn_layer = Flux.flatten
        end
        push!(nn_layers, nn_layer)
    end

    # chain loaded layers together
    nn = Chain([nn_layers]...)
    
    return nn
end


function dense_layer_from_config(layer_config::Dict)

    # activation (identity by default)
    if haskey(layer_config, "activation")
        act_str = layer_config["activation"]
        act_fn = haskey(activation_fns_lookup, act_str) ? activation_fns_lookup[act_str] : identity
    else
        act_fn = identity
    end

    # bias (true by default)
    bias = haskey(layer_config, "bias") ? layer_config["bias"] : true

    # init (glorot_uniform default)
    if haskey(layer_config, "init")
        init_str = layer_config["init"]
        init_fn = haskey(initialization_fns_lookup, init_str) ? initialization_fns_lookup[init_str] : Flux.glorot_uniform
    else
        init_fn = Flux.glorot_uniform
    end

    return Dense(layer_config["in"]=>layer_config["out"], act_fn; init=init_fn, bias=bias)
end


function conv_layer_from_config(layer_config::Dict)

    # filter
    filter = tuple(layer_config["filter"]...)

    # stride can be int or tuple (per-dimension)
    if haskey(layer_config, "stride")
        if isa(layer_config["stride"], Array)  # tuple
            stride = tuple(layer_config["stride"]...)
        else  # int
            stride = layer_config["stride"]
        end
    else
        stride = 1
    end

    # dilation can be int or tuple (per-dimension)
    if haskey(layer_config, "stride")
        if isa(layer_config["stride"], Array)  # tuple
            stride = tuple(layer_config["stride"]...)
        else  # int
            stride = layer_config["stride"]
        end
    else
        stride = window
    end


    # bias (true by default)
    bias = haskey(layer_config, "bias") ? layer_config["bias"] : true

    # init (glorot_uniform default)
    if haskey(layer_config, "init")
        init_str = layer_config["init"]
        init_fn = haskey(initialization_fns_lookup, init_str) ? initialization_fns_lookup[init_str] : Flux.glorot_uniform
    else
        init_fn = Flux.glorot_uniform
    end

    return Conv(filter, layer_config["in"]=>layer_config["out"];
                stride=parse_tuple_or_int_hyperparam(layer_config, "stride", 1),  # NTuple or Int
                pad=haskey(layer_config, "pad") ? layer_config["pad"] : 0,  # Int
                dilation=parse_tuple_or_int_hyperparam(layer_config, "dilation", 1),  # NTuple or Int
                groups=haskey(layer_config, "groups") ? layer_config["groups"] : 1,
                bias=bias,
                init=init_fn)

end


function max_pool_layer_from_config(layer_config::Dict)

    window = tuple(layer_config["window"]...)  # NTuple
    pad = haskey(layer_config, "pad") ? layer_config["pad"] : 0  # Int
    stride = parse_tuple_or_int_hyperparam(layer_config, "stride", window)  # NTuple/Int

    return MaxPool(window;
                   pad=pad,
                   stride=stride)
end


function fourier_operator_layer_from_config(layer_config::Dict)
    return nothing
end


function MLP(; inputsize=(3600,1))
    """
    https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl
    """

    return Chain( Dense(prod(inputsize), 256, relu),
                Dense(256, 62*62))
end


function FourierNeuralOperator()
    
    # return NeuralOperators.FourierNeuralOperator(;
    #                   ch = (3600, 64, 64, 64, 64, 64, 128, 62*62),
    #                   modes = (16, ),
    #                   σ = Flux.gelu
    # )

    modes = (16, )
    ch = 64 => 64
    σ = Flux.gelu

    return Chain(
        # # operator projects data between infinite-dimensional spaces
        OperatorKernel(60 => 60, (16, ), FourierTransform, Flux.gelu, permuted=false),
        # FourierOperator(ch, modes, σ),
        # FourierOperator(ch, modes, σ),
        # FourierOperator(ch, modes),
        # # project infinite-dimensional function to finite-dimensional space
        Dense(60, 128, σ),
        Dense(128, 62*62)#,
        # flatten
    )

end

function CNN(; inputsize=(60,60,1), outputsize=62*62) 
    """
    https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl
    """

    out_conv_size = (inputsize[1]÷4 - 3, inputsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), inputsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, outputsize)
          )
end


function parse_tuple_or_int_hyperparam(layer_config, hyperparam_str::String, default)
    
    # hyperparam can be int or tuple (per-dimension), helper function especially for parsing CNN configs
    if haskey(layer_config, hyperparam_str)
        if isa(layer_config[hyperparam_str], Array)  # tuple
            hyperparam = tuple(layer_config[hyperparam_str]...)
        else  # int
            hyperparam = layer_config[hyperparam_str]
        end
    else
        hyperparam = default
    end

    return hyperparam

end    


if abspath(PROGRAM_FILE) == @__FILE__
    # println("try loading mlp config")
    # load_model_from_json("/nfs/raid66/u11/users/brozonoy-ad/Oceananigans.jl/NeuralOceans.jl/src/configs/mlp.json")
    println("try loading cnn config")
    load_model_from_json("/nfs/raid66/u11/users/brozonoy-ad/Oceananigans.jl/NeuralOceans.jl/src/configs/cnn.json")
end