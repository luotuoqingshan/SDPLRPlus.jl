"""
    barvinok_pataki(n, m)

Compute the barvinok-pataki bound min{n, sqrt{2m} + 1}
for a SDP problem where X has size n x n and there are m constraints.
"""
function barvinok_pataki(n::T, m::T) where {T<:Int}
    return min(n, T(floor(sqrt(2*m)+1)))
end

barvinok_pataki(data::SDPData) = barvinok_pataki(data.n, data.m)
