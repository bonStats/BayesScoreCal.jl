mutable struct UnivariateAffine <: Transform{1}
    σ::Real
    b::Real
end

function UnivariateAffine()
    UnivariateAffine(0.0,0.0)
end

dimension(::UnivariateAffine) = () # object dimension
length(::UnivariateAffine) = 1 # for internal use (within CholeskyAffine functions)
nparam(::UnivariateAffine) = 2 # number of params

function update!(uaf::UnivariateAffine, logσ::Real, b::Real)
    uaf.σ = exp(logσ)
    uaf.b = b
    return uaf
end

function update!(uaf::UnivariateAffine, v::Vector{<:Real})
    uaf.σ = exp(v[1])
    uaf.b = v[2]
    return uaf
end

scaleparamvec(tf::UnivariateAffine) = log(tf.σ)
biasparamvec(tf::UnivariateAffine) = tf.b
paramvec(tf::UnivariateAffine) = [tf.σ; tf.b]

(uaf::UnivariateAffine)(x::T, μs::T) where {T<:Real} = uaf.σ * (x - μs) + μs + uaf.b 

(uaf::UnivariateAffine)(cal::Calibration{T}) where {T<:Real} = Calibration(cal.values, hcat([uaf.(clm, [cal.μs[i]]) for (i, clm) in enumerate(eachcol(cal.samples))]...))
