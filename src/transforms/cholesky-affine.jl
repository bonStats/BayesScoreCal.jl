include("pd-chol-helper.jl")

mutable struct CholeskyAffine{N} <: Transform{N}
    L::LowerTriangular{Real, Matrix{Real}}
    b::Vector{Real}
    function CholeskyAffine(L::LowerTriangular{<:Real, Matrix{<:Real}}, b::Vector{<:Real})
        @assert length(b) == size(L, 1)
        new{length(b)}(L, b)
    end
end

CholeskyAffine(vL::Vector{<:Real}, b::Vector{<:Real}) = CholeskyAffine(vec2chol(vL), b)
CholeskyAffine(vLb::Vector{<:Real}, d::Int64) = CholeskyAffine(vLb[1:(end-d)], vLb[(end-d+1):end])
function CholeskyAffine(d::Int64)
    par_zero = zeros(Int((d^2 + d)/2 + d))
    CholeskyAffine(par_zero, d)
end

dimension(::CholeskyAffine{N}) where {N} = (N,) # object dimension
length(::CholeskyAffine{N}) where {N} = N # for internal use (within CholeskyAffine functions)
nparam(::CholeskyAffine{N}) where {N} = N*(N + 3)/2 # number of params

function update!(chaf::CholeskyAffine, vL::Vector{<:Real}, b::Vector{<:Real})
    chaf.L .= vec2chol(vL)
    chaf.b .= b
    return chaf
end

function update!(chaf::CholeskyAffine, vLb::Vector{<:Real})
    n = length(chaf)
    chaf.L .= vec2chol(vLb[1:(end-n)])
    chaf.b .= vLb[(end-n+1):end]
    return chaf
end

scaleparamvec(tf::CholeskyAffine) = chol2vec(tf.L)
biasparamvec(tf::CholeskyAffine) = tf.b
paramvec(tf::CholeskyAffine) = [chol2vec(tf.L); tf.b]

(chaf::CholeskyAffine)(x::T, μs::T) where {T} = chaf.L * (x - μs) + μs + chaf.b 

(chaf::CholeskyAffine)(cal::Calibration{T}) where {T} = Calibration(cal.values, hcat([chaf.(clm, [cal.μs[i]]) for (i, clm) in enumerate(eachcol(cal.samples))]...))
