using BayesScoreCal
using Turing
using Turing: Variational
using Distributions
using LinearAlgebra
using SharedArrays
using JLD2

# helper
vrand(dist,N) = [rand(dist) for i in 1:N]
multiplyvariance(x::Vector{T},v::T,μ::T) where {T} = [Diagonal(v)] .* (x .- [μ]) .+ [μ]

# data setup
d = 2
n = 200
trueμ = rand(Normal(), d)
crossprod(X::Matrix) = X' * X
Σ = crossprod(rand(2,2))

datagen(μ,n) = [rand(MvNormal(μ, Σ)) for i in 1:n]
y = datagen(trueμ,n)

@model function model(y, d = length(y[1]))
    μ ~ MvNormal(zeros(d), I)
    for i in eachindex(y)
        y[i] ~ MvNormal(μ, Σ)
    end
end

mod = model(y)

# BSC settings
N_samples = 2000
N_importance = 100
N_energy = 1000
vmultiplier = 2.0

# artificial bias
bbias = 1/10

# approximate posterior for data y
advi = ADVI(10, 1000)
q_approx = vi(mod, advi)

# μ ∈ ℝᵈ so no transformation/bijectors needed
approx_samples = vrand(q_approx, N_samples) .+ [repeat([bbias], d)]
μ_approx_samples = mean(approx_samples)

# calibration points/importance sample: increase variance if required via vmultiplier >= 1
cal_points = multiplyvariance(sample(approx_samples, N_importance), vmultiplier*ones(d), μ_approx_samples)

# newx approx model samples: pre-allocate
approx_samples_newy = SharedArray{Float64}(length(cal_points[1]), N_energy, N_importance)

# how many threads can we parallelise approximate posteriors over?
Threads.nthreads()

# generate new data/models: cal_point -> new y -> new mod -> new approx posterior
Threads.@threads for j in 1:length(cal_points)
    println("Working on... $j") # need to supress vi progress meter and info

    # new data
    newy = datagen(cal_points[j], n)
    # (model|new data)
    mod_newy = model(newy)
    # vi(model|new data)
    q_newy = vi(mod_newy, advi)
    # samples from vi(model|new data)
    approx_samples_newy[:,:,j] = rand(q_newy, N_energy) .+ bbias          
    
end

# resize
approx_samples_newy = [approx_samples_newy[:,i,j] for i in 1:N_energy, j in 1:N_importance]

# calibration object
cal = Calibration(cal_points, approx_samples_newy)

# use cache...
#save_object("examples/simple-vi/simple-vi-cache.jld2", (cal = cal, trueμ = trueμ, y = y))
use_cache = false
if use_cache
    cal, trueμ, y = load_object("examples/simple-vi/simple-vi-cache.jld2")
    mod = model(y)
    q_approx = vi(mod, advi)
    approx_samples = vrand(q_approx, N_samples) .+ [repeat([bbias], d)]
    μ_approx_samples = mean(approx_samples)
end

# weights = 1 is good for large enough data
w = ones(N_importance)

tf = CholeskyAffine(d)
res = energyscorecalibrate!(tf, cal, w)

# original
coverage(cal, [0.5, 0.8, 0.9, 0.95])
rmse(cal)
bias(cal)

# transformed
coverage(cal, tf, [0.5, 0.8, 0.9, 0.95])
rmse(cal, tf)
bias(cal, tf)

# look at learned transformation
tf.b
tf.L

adjust_samples = tf.(approx_samples, [μ_approx_samples])


using StatsPlots

samplesμ1 = [getindex.(approx_samples,1) getindex.(adjust_samples,1)]
samplesμ2 = [getindex.(approx_samples,2) getindex.(adjust_samples,2)]


density(samplesμ1, labels = ["approx" "adjust"])
vline!([trueμ[1]], labels = "true value")

density(samplesμ2, labels = ["approx" "adjust"])
vline!([trueμ[2]], labels = "true value")
