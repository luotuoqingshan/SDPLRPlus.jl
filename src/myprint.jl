function printheading(start)
    if start == 1 # print start heading
        println("="^121)
        println((" "^40)*"SDPLR.jl: A julia reimplementation of SDPLR"*(" "^30))
        println("="^121)
    else
        println("="^121)
        println(" "^54*"End of SDPLR.jl"*" "^20)
        println("="^121)
    end
end

function printintermediate(
    majoriter::Int,
    localiter::Int,
    iter::Int,
    𝓛_val::Float64,
    obj::Float64,
    stationarity::Float64,
    primal_vio::Float64,
    dual_bound::Float64,
)
    println("Major Iter"*(" "^3)*"Local Iter"*(" "^3)*"Total Iter"*(" "^3)
            *"Lagrangian Val"*(" "^3)*"Objective Val"*(" "^3)*"Stationarity"*(" "^3)
            *"Primal Feasibility"*(" "^3)*"Duality Bound")
    @printf("%10d   %10d   %10d   %14e   %13e   %12e   %18e   %13e\n",
            majoriter, localiter, iter, 𝓛_val,
            obj, stationarity, primal_vio, dual_bound)
end

#printheading(1)
#printintermediate(1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)
#printheading(0)

#printheading(1)
