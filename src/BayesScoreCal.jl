module BayesScoreCal

import LinearAlgebra: LowerTriangular, norm, tril!, I, Diagonal, UniformScaling
import Statistics: mean, quantile
import Random: shuffle
import Base: length, size
import Intervals: Interval

using Optim

include("abstract-structs.jl")
include("calibration.jl")
include("energy-score.jl")
include("diagnostics.jl")
include("transforms/cholesky-affine.jl")


export Transform
export Calibration
export CholeskyAffine
export energyscorecalibrate!
export coverage, rmse, bias

end
