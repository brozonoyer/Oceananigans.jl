using NeuralOceans: train, decode
using BSON
using BSON: @save, @load
using ArgParse


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--data", "-i"
            help = "path to jld2 in jld2 format"
            arg_type = String
            required = true
        "--model_config", "-c"
            help = "model architecture config"
            arg_type = String
            required = true
        "--model", "-o"
            help = "path to save model"
            arg_type = String
            required = true
        "--decode", "-d"
            help = "do inference rather than training"
            action = :store_true
        "--fourier", "-f"
            help = "whether model trains/decodes in fourier space"
            action = :store_true
        "--cnn_input"
            help = "if training CNN, input data can be in 2d format, otherwise it is flattened"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__

    parsed_args = parse_commandline()
    if parsed_args["decode"]
        decode(parsed_args["model"]; model_config=parsed_args["model_config"], data=parsed_args["data"], fourier=parsed_args["fourier"])
    else
        model = train(; model_config=parsed_args["model_config"], data=parsed_args["data"], cnn_input=parsed_args["cnn_input"], fourier=parsed_args["fourier"])
        @save parsed_args["model"] model
    end

end
