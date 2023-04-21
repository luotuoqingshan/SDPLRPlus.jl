function printheading(start)
    if start == 1 # print start heading
        println("="^60)
        println((" "^10)*"SDPLR.jl: A julia reimplementation of SDPLR"*(" "^10))
        println("="^60)
    else
        println("="^60)
        println(" "^22*"End of SDPLR.jl"*" "^20)
        println("="^60)
    end
end

function printintermediate()
end


#printheading(1)
