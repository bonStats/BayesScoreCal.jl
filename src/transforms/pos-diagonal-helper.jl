# Functions for converting
# unconstrained vector v ∈ ℜᵈ <---> M where diag(M) > 0

function vec2matposdiag(v::Vector{<:Real})
    d = Int(sqrt(length(v)))
    V = reshape(v, d, d)
    V[diagind(V)] .= exp.(V[diagind(V)]) 
    return V
end

function matposdiag2vec(V::Matrix{<:Real})
    V2 = copy(V)
    V2[diagind(V)] .= log.(V2[diagind(V)])
    return reshape(V2,Int(d^2))
end