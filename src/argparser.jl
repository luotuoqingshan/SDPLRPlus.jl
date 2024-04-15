using ArgParse

function argparser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--ptol"
            help = "primal infeasiblity tolerance"
            arg_type = Float64
            default = 1e-2
        "--objtol"
            help = "duality gap tolerance"
            arg_type = Float64
            default = 1e-2
        "--maxtime"
            help = "Limit on running time(seconds)."
            arg_type = Float64
            default = 3600.0
    end
    return parse_args(s)
end