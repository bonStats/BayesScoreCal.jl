
# helper
function credprob2ends(p::Float64) 
    lp = (1 - p)/2
    up = 1 - lp
    [lp, up]
end

vec2interval(x::Vector{<:Real}) = Interval(x[1], x[2])

vquantile(x::AbstractVector{T}, p::Vector{<:Real}) where {T<:Vector}= [vec2interval(quantile(getindex.(x, i), p)) for i in eachindex(x[1])]

function vquantin(x::AbstractVector{T}, p::Real, v::T) where T
    v .∈ vquantile(x, credprob2ends(p)) 
end

# calibration coverage check

function coverage(cal::Calibration, prob::AbstractVector{<:Real})
    calin = [vquantin(clmj, prob[i], cal.values[j]) for i in eachindex(prob), (j, clmj) in enumerate(eachcol(cal.samples))]
    vec( sum(calin, dims = 2) ./ size(calin, 2) )
end

coverage(cal::Calibration, tf::Transform, prob::AbstractVector{<:Real}) = coverage(tf(cal), prob)

# rmse

sumsquares(x::Vector{<:Real}) = sum(x .^ 2)

function rmse(cal::Calibration)

    meanss = [mean(sumsquares.(clmj .- [cal.values[j]])) for (j, clmj) in enumerate(eachcol(cal.samples))]
    
    mean(sqrt.(meanss))
end

rmse(cal::Calibration, tf::Transform) = rmse(tf(cal))

#bias 

function bias(cal::Calibration)

    mean([mean(clmj .- [cal.values[j]]) for (j, clmj) in enumerate(eachcol(cal.samples))])
    
end

bias(cal::Calibration, tf::Transform) = bias(tf(cal))
