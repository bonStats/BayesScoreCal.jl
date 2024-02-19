struct PermuteVector
    x::Vector{Int64}
    """
    PermuteVector(n::Int64)

    Creates random permutation vector of length n
    """
    PermuteVector(n::Int64) = new(shuffle(1:n)) 
end

function score(sample::Vector{Vector{S}}, value::Vector{T}, perm::PermuteVector, β::Real) where {S <: Real, T<: Real}
    mean(norm.(sample .- [value]) .^ β) - 0.5 * mean(norm.(sample .- sample[perm.x]) .^ β)
end
score(sample::Vector{Vector{S}}, value::Vector{T}, perm::PermuteVector, β::Real, sM::AbstractMatrix{T}) where {S <: Real, T<: Real} = score([sM] .* sample, sM * value, perm, β)
score(sample::Vector{Vector{S}}, value::Vector{T}, perm::PermuteVector, β::Real, sM::UniformScaling{Bool}) where {S <: Real, T<: Real} = score(sample, value, perm, β)

# scalars
function score(sample::Vector{S}, value::T, perm::PermuteVector, β::Real) where {S <: Real, T<: Real}
    mean(norm.(sample .- value) .^ β) - 0.5 * mean(norm.(sample - sample[perm.x]) .^ β)
end
score(sample::Vector{S}, value::T, perm::PermuteVector, β::Real, s::T) where {S <: Real, T<: Real} = score(s .* sample, s * value, perm, β)
score(sample::Vector{S}, value::T, perm::PermuteVector, β::Real, s::UniformScaling{Bool}) where {S <: Real, T<: Real} = score(sample, value, perm, β)

function negenergyscore(tf::Transform, cal::Calibration, weights::Vector{Float64}, perm::PermuteVector, β::Real, sM::Union{AbstractMatrix,UniformScaling}, penalty::Tuple{Float64, Float64, Float64})
    
    nesval = 0.0

    for j in 1:size(cal.samples,2) # iterate over columns/calibration sets
        # iterate over samples
        tfsample = [tf(cal.samples[i,j], cal.μs[j]) for i in 1:size(cal.samples,1)]
        nesval += weights[j] * score(tfsample, cal.values[j], perm, β, sM)
    end

    return nesval + penalty[1] * sum(biasparamvec(tf) .^2) + penalty[2] * sum(scaleparamvec(tf) .^ 2) + penalty[3] * sum(covparamvec(tf) .^ 2) 
    
end

"""
    energyscorecalibrate!(tf::T, cal::Calibration, weights::Vector{Float64}; β::Real = 1.0, scaling = I, penalty::Tuple{Float64,Float64,Float64} = (0.0,0.0,0.0), options::Optim.Options = Optim.Options()) where {T<:Transform}

Finds parameters of transform `tf` using Bayesian Score Calibration.

# Arguments
- `tf<:Transform`: instance of transformation, updated with optimal parameters.
- `cal::Calibration`: Calibration object holding samples and true values.
- `weights::Vector{Float64}```: Importance weights, commonly `w = ones(length(cal))`.

# Optional arguments
- `β::Real = 1.0`: Energy score parameter β ∈ (0,2).
- `scaling = I`: Scaling matrix to rescale parameters.
- `penalty::Tuple{Float64,Float64,Float64} = (0.0,0.0,0.0)`: Penalty terms for (bias, variance, covariance [off-diagonal]).
- `options::Optim.Options = Optim.Options()`: Options to give to `Optim`.

"""
function energyscorecalibrate!(tf::T, cal::Calibration, weights::Vector{Float64}; β::Real = 1.0, scaling = I, penalty::Tuple{Float64,Float64,Float64} = (0.0,0.0,0.0), options::Optim.Options = Optim.Options()) where {T<:Transform}

    @assert dimension(tf) == dimension(cal)

    perm = PermuteVector(nsamples(cal))
    initialx = paramvec(tf)

    f = OnceDifferentiable((x) -> negenergyscore(maketransform(tf,x), cal, weights, perm, β, scaling, penalty), initialx; autodiff = :forward)

    res = Optim.optimize(f, initialx, BFGS(), options)
    
    update!(tf, Optim.minimizer(res))

    return res

end