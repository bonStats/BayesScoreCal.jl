include("pos-diagonal-helper.jl")
# NOTE: EigenAffine does NOT constrain VV' = I, this is controlled by corr penalty in optimisation

mutable struct EigenAffine{N,T<:Real} <: Transform{N}
    d::Vector{T} # d = √D
    V::Matrix{T}
    b::Vector{T}
    function EigenAffine(d::Vector{T}, V::Matrix{T}, b::Vector{T}) where {T<:Real}
        @assert length(b) == length(d) == size(V,1) == size(V,2)
        new{length(b),T}(d, V, b)
    end
end

EigenAffine(v::Vector{<:Real}, d::Int64) = EigenAffine(exp.(v[1:d]), vec2matposdiag(v[(d+1):(end-d)]), v[(end-d+1):end])
function EigenAffine(d::Int64)
    par_zero = zeros(Int( d * (d + 2) ))
    EigenAffine(par_zero, d)
end


dimension(::EigenAffine{N,T}) where {N,T} = (N,) # object dimension
length(::EigenAffine{N,T}) where {N,T} = N # for internal use (within EigenAffine functions)
nparam(::EigenAffine{N,T}) where {N,T} = N * (N + 2) # number of params

# for update! EigenAffine may not have same type as other arguments (e.g. Float64 -> Dual)
function update!(tf::EigenAffine{N,T}, d::Vector{T}, V::Matrix{T}, b::Vector{T}) where {N,T<:Real} 
    tf.d .= d
    tf.V .= V
    tf.b .= b
end

update!(tf::EigenAffine{N,T}, v::Vector{T}) where {N,T<:Real} = update!(tf, exp.(v[1:d]), vec2matposdiag(v[(d+1):(end-d)]), v[(end-d+1):end])

maketransform(::EigenAffine{N,Ti}, v::Vector{To}) where {N,Ti<:Real,To<:Real} = EigenAffine(v, N)

idpenalty(tf::EigenAffine) = sum((tf.V * tf.V' - I) .^ 2) # for identifability of parameters
corrpenalty(tf::EigenAffine) = sum((tf.V - I) .^ 2)
scalepenalty(tf::EigenAffine) = sum(tf.d .^ 2)

paramvec(tf::EigenAffine) = [log.(tf.d); matposdiag2vec(tf.V); tf.b]

# for  (::EigenAffine) may not have same type as input (e.g. Float64 -> Dual)
(tf::EigenAffine)(x::T, μs::T) where {T} = tf.V * ( tf.d .* (x - μs)) + μs + tf.b 