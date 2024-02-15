include("pd-chol-helper.jl")

mutable struct CholeskyAffine{N,T<:Real} <: Transform{N}
    L::LowerTriangular{T, Matrix{T}}
    b::Vector{T}
    function CholeskyAffine(L::LowerTriangular{T, Matrix{T}}, b::Vector{T}) where {T<:Real}
        @assert length(b) == size(L, 1)
        new{length(b),T}(L, b)
    end
end

CholeskyAffine(vL::Vector{T}, b::Vector{T}) where {T<:Real}  = CholeskyAffine(vec2chol(vL), b)
CholeskyAffine(vLb::Vector{<:Real}, d::Int64) = CholeskyAffine(vLb[1:(end-d)], vLb[(end-d+1):end])
function CholeskyAffine(d::Int64)
    par_zero = zeros(Int((d^2 + d)/2 + d))
    CholeskyAffine(par_zero, d)
end

dimension(::CholeskyAffine{N,T}) where {N,T} = (N,) # object dimension
length(::CholeskyAffine{N,T}) where {N,T} = N # for internal use (within CholeskyAffine functions)
nparam(::CholeskyAffine{N,T}) where {N,T} = N*(N + 3)/2 # number of params

# for update! CholeskyAffine may not have same type as other arguments (e.g. Float64 -> Dual)
function update!(chaf::CholeskyAffine, vL::Vector{T}, b::Vector{T}) where {T<:Real} 
    chaf = CholeskyAffine(vec2chol(vL), b)
    return chaf
end

function update!(chaf::CholeskyAffine, vLb::Vector{T}) where {T<:Real} 
    n = length(chaf)
    return update!(chaf, vLb[1:(end-n)], vLb[(end-n+1):end])
end

scaleparamvec(tf::CholeskyAffine) = chol2vec(tf.L)
biasparamvec(tf::CholeskyAffine) = tf.b
paramvec(tf::CholeskyAffine) = [chol2vec(tf.L); tf.b]

# for  (::CholeskyAffine) may not have same type as input (e.g. Float64 -> Dual)
(chaf::CholeskyAffine)(x::T, μs::T) where {T} = chaf.L * (x - μs) + μs + chaf.b 

(chaf::CholeskyAffine)(cal::Calibration{T}) where {T} = Calibration(cal.values, hcat([chaf.(clm, [cal.μs[i]]) for (i, clm) in enumerate(eachcol(cal.samples))]...))
