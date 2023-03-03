# Calibration object

vecif(x::Matrix) = (size(x, 1) == 1) | (size(x, 2) == 1) ? vec(x) : x

struct Calibration{T}
    values::Vector{T} # true data-generating values
    samples::Matrix{T} # approximate samples
    μs::Vector{T}
    function Calibration(values::Vector{T}, samples::Matrix{T}) where {T}
        @assert length(values) == size(samples, 2) # rows samples, columns datasets
        μs = vecif(mean(samples, dims=1))
        new{T}(values, samples, μs)
    end
end

Calibration(values::Vector{T}, samples::Matrix{T}, parids::Vector{Int64}) where {T} = Calibration(getindex.(values, [parids]), getindex.(samples, [parids]))

"""
    length(cal::Calibration)

Number of calibration (importance) samples (repeated parameter-data pairs). 
"""
length(cal::Calibration) = length(cal.values)




"""
    dimension(cal::Calibration)

Size/dimension of parameters to be calibrated.
"""
dimension(cal::Calibration) = size(cal.values[1])
size(cal::Calibration) = dimension(cal)


"""
    nsamples(cal::Calibration)

Number of samples used for each distribution represented in calibration object.
"""
nsamples(cal::Calibration) = size(cal.samples,1)



"""
    uniformscaling(cal::Calibration)

Generate UniformScaling object for calibration.
"""
uniformscaling(cal::Calibration) = I(dimension(cal)[1])
