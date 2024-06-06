function printheading(start)
    if start == 1 # print start heading
        println("="^121)
        println((" "^29)*"SDPLR.jl: a julia implementation of SDPLR with objval gap bound"*(" "^29))
        println("="^121)
    else
        println("="^121)
        println(" "^54*"End of SDPLR.jl"*" "^20)
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
    grad_norm::Float64,
    primal_vio_norm::Float64,
    dual_bound::Float64,
)
    @printf("%12s  %10s  %10s  %10s  %12s  %12s  %12s  %12s  %12s\n",
            "dataset", "majoriter", "localiter", "totaliter", "Lagranval",
            "objval", "gradnorm", "pvio val", "best suboptimality")
    @printf("%12s %10d  %10d  %10d  %12.3e  %12.3e  %12.3e  %12.3e  %12.3e\n",
            dataset, majoriter, localiter, iter, 𝓛_val,
            obj, grad_norm, primal_vio_norm, dual_bound)
end

