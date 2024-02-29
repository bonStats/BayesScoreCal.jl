mutable struct UnivariateAffine{T<:Real} <: Transform{1}
    σ::T
    b::T
end

function UnivariateAffine(v::Vector{T}) where {T<:Real}
    UnivariateAffine(exp(v[1]),v[2])
end

function UnivariateAffine()
    UnivariateAffine(0.0,0.0)
end

dimension(::UnivariateAffine) = () # object dimension
length(::UnivariateAffine) = 1 # for internal use (within CholeskyAffine functions)
nparam(::UnivariateAffine) = 2 # number of params

function update!(uaf::UnivariateAffine{T}, logσ::T, b::T) where {T<:Real}
    uaf.σ = exp(logσ)
    uaf.b = b
    return uaf
end

update!(uaf::UnivariateAffine{T}, v::Vector{T}) where {T<:Real} = update!(uaf, exp(v[1]), v[2])

maketransform(::UnivariateAffine{Ti}, v::Vector{To}) where {Ti<:Real,To<:Real} = UnivariateAffine(v)

idpenalty(tf::UnivariateAffine) = 0.0
corrpenalty(tf::UnivariateAffine) = 0.0
scalepenalty(tf::UnivariateAffine) = log(tf.σ) ^ 2

paramvec(tf::UnivariateAffine) = [tf.σ; tf.b]

(uaf::UnivariateAffine)(x::T, μs::T) where {T<:Real} = uaf.σ * (x - μs) + μs + uaf.b