(tf::Transform{N})(cal::Calibration{T}) where {N,T} = Calibration(cal.values, hcat([tf.(clm, [cal.μs[i]]) for (i, clm) in enumerate(eachcol(cal.samples))]...))
