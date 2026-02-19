function printheading(start)
    if start == 1 # print start heading
        println("="^121)
        println(
            (" "^29)*"SDPLRPlus.jl: a julia implementation of SDPLR with objval gap bound"*(
                " "^29
            ),
        )
        println("="^121)
    else
        println("="^121)
        println(" "^54*"End of SDPLRPlus.jl"*" "^20)
        println("="^121)
    end
end

function printintermediate(
    dataset::String,
    majoriter::Int,
    localiter::Int,
    iter::Int,
    𝓛_val::Float64,
    obj::Float64,
    σ::Float64,
    cur_gtol::Float64,
    cur_ptol::Float64,
    grad_norm::Float64,
    primal_vio_norm::Float64,
    dual_bound::Float64,
    max_dual_value::Float64,
)
    header = [
        "dataset",
        "T",
        "Iterₜ",
        "TotIter",
        "ℒ",
        "pobj",
        "σ",
        "ηₜ",
        "ωₜ",
        "‖grad‖",
        "‖pinfeas‖",
        "min pobj-dobj",
        "max dobj",
    ]
    data =
        [dataset majoriter localiter iter 𝓛_val obj σ cur_gtol cur_ptol grad_norm primal_vio_norm dual_bound max_dual_value]
    pretty_table(
        data;
        column_labels=header,
        formatters=[
            fmt__printf("%s", [1]),
            fmt__printf("%d", 2:4),
            fmt__printf("%.2E", 5:13),
        ],
    )
end
