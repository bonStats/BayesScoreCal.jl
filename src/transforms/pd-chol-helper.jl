# Functions for converting
# unconstrained vector v ∈ ℜᵈ <---> L 
# where L is lower triangular has diagonal elements and L'L ∈ PD(n,n)

"""
    expdiag(v::Real, i::Int64, j::Int64)

Exponentiate the diagonal based on row-column (i,j) location.
"""
expdiag(v::Real, i::Int64, j::Int64) = i==j ? exp(v) : v 


"""
    vec2chol(v::Vector{<:Real}, n::Int64)

Convert a unconstrained vector v to lower-triangular cholesky decomposition of (n x n) positive definite matrix.
"""
function vec2chol(v::Vector{T}, n::Int64) where {T<:Real}
    k = 0
    LowerTriangular([ j<=i ? (k+=1; expdiag(v[k], i , j)) : zero(T) for i=1:n, j=1:n ])
end

vec2chol(v::Vector{<:Real}) = vec2chol(v, cholsize(v))




"""
    cholsize(v::Vector{<:Real})

Get size of cholesky matrix vector v.
"""
cholsize(v::Vector{<:Real}) = Int( (sqrt(8 * length(v) + 1) - 1)/2 )




"""
    vecsize(L::LowerTriangular)

Get size of vector for cholesky matrix L.
"""
vecsize(L::LowerTriangular) = Int( size(L,1) * (size(L,1) + 1) / 2 )




"""
    chol2vec(L::LowerTriangular)

Convert a lower-triangular cholesky decomposition to unconstrained vector v. 
"""
chol2vec(L::LowerTriangular) = vcat( [[log(col[i]);col[(i+1):end]] for (i, col) in enumerate(eachcol(L))]... )

"""
    chol2varvec(L::LowerTriangular)

Convert a lower-triangular cholesky decomposition to the diagonal unconstrained vector v. 
"""
chol2varvec(L::LowerTriangular) = [log(col[i]) for (i, col) in enumerate(eachcol(L))]


"""
    chol2corrvec(L::LowerTriangular)

Convert a lower-triangular cholesky decomposition to the off diagonal unconstrained vector v of CORRELATIONS. 
"""
chol2covvec(L::LowerTriangular) = vcat( [col[(i+1):end] for (i, col) in enumerate(eachcol(L))]... )
