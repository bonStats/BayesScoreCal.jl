module BayesScoreCal

import LinearAlgebra: LowerTriangular, norm, tril!, I, Diagonal, UniformScaling, diagind
import Statistics: mean, quantile
import Random: shuffle
import Base: length, size
import Intervals: Interval

using Optim

include("abstract-structs.jl")
include("calibration.jl")
include("energy-score.jl")
include("diagnostics.jl")
include("transforms/abstract-transform.jl")
include("transforms/eigen-affine.jl")
include("transforms/cholesky-affine.jl")
include("transforms/uni-affine.jl")

export Transform
export Calibration
export EigenAffine
export CholeskyAffine
export UnivariateAffine
export energyscorecalibrate!
export coverage, rmse, bias

end
