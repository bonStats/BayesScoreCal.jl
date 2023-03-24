using Plots
using DifferentialEquations
using LinearAlgebra
using Distributions
using Turing
using StatsPlots
using JLD2
using Distributed
using SharedArrays
using DataFrames
using CSV
using BayesScoreCal

# set to redo simulations (CACHE TOO LARGE TO ADD TO REPO)
use_cache = false
if use_cache
    data_cache = load_object("lotka-volterra-cache.jld2")

    data = data_cache.data
    cal_points = data_cache.cal_p
    tr_approx_samples_newx = data_cache.tr_cal_s
end


# approximate/true model settings
N_samples = 1000
n_adapt = 1000

# optimisation settings
N_importance = 200
N_energy = 1000
energyβ = 1.0
vmultiplier = 2.0

checkprobs = range(0.1,0.95,step=0.05)


# Turing helpers
function getsamples(chains::Chains, sym::Symbol, sample_chain::Union{Iterators.ProductIterator{Tuple{UnitRange{Int64}, UnitRange{Int64}}},Vector{Tuple{Int64, Int64}}})
    smpls = [vec(chains[sc[1], namesingroup(chains, sym), sc[2]].value) for sc in sample_chain]
    convert.(Vector{Float64}, smpls)
end
function getsamples(chains::Chains, sym::Symbol, N::Int64)
    # down sample
    sample_chain = sample([Iterators.product(1:length(chains), 1:size(chains, 3))...], N, replace = false)
    getsamples(chains, sym, sample_chain)
end
function getsamples(chains::Chains, sym::Symbol)
    # full sample
    sample_chain = Iterators.product(1:length(chains), 1:size(chains, 3))
    getsamples(chains, sym, sample_chain)
end
getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))

# setup Lotka Volterra
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
function multiplicative_noise!(du, u, p, t)
    x, y = u
    du[1] = p[5] * x
    return du[2] = p[6] * y
end

true_p = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = dx = α * x - β * x * y
    return du[2] = dy = δ * x * y - γ * y
end

prob_sde = SDEProblem(lotka_volterra!, multiplicative_noise!, u0, tspan, true_p)

# simulate data
if !use_cache
    data = solve(prob_sde, SOSRI(), saveat=0.1)
    plot(data)

    # example of approximate data
    approx_data = solve(prob_sde, EM(); p=p, saveat=0.1, dt = 0.01)
    plot!(approx_data)
end

@model function fitlv_sde(data, prob)

    pars = zeros(4)
    # Prior distributions.
    pars[1] ~ Uniform(0.1,5) #α
    pars[2] ~ Uniform(0.1,5) #β
    pars[3] ~ Uniform(0.1,5) #γ
    pars[4] ~ Uniform(0.1,5) #δ
    ϕ1 = 0.1
    ϕ2 = 0.1

    σ ~ InverseGamma(2, 3)

    # Simulate stochastic Lotka-Volterra model.
    predicted = solve(prob, EM(); p=[pars..., ϕ1, ϕ2], saveat=0.1, dt = 0.01)
    #predicted = solve(prob, SOSRI(); p=[pars..., ϕ1, ϕ2], saveat=0.1)

    # Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        println("bad solver")
        Turing.@addlogprob! -Inf
        return nothing
    end

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model_sde = fitlv_sde(data, prob_sde)

initial_par = [1.0, 1.0, 1.0, 1.0]
initial_other = [1.0]
intial_all = vcat(initial_par,initial_other)

ch_approx = sample(model_sde, NUTS(0.25), N_samples; init_params=intial_all, progress=true)

mean(getsamples(ch_approx,  :pars))
std(getsamples(ch_approx,  :pars))

corner(ch_approx)

## Calibration

function multiplyscale(x::Vector{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

# transform
bij_all = bijector(model_sde)
bij = Bijectors.stack(bij_all.bs[1:length(initial_par)]...)

# select and transform approx samples
select_approx_samples = getsamples(ch_approx, :pars, N_importance)

# sample calibration points
if !use_cache
    cal_points = inv(bij).(multiplyscale(bij.(select_approx_samples), vmultiplier))

    # newx approx models: pre-allocate
    tr_approx_samples_newx = SharedArray{Float64}(length(cal_points[1]), N_energy, N_importance)

    pmeter = Progress(length(cal_points))

    Threads.@threads for t in 1:length(cal_points)
        # new data
        new_prob = remake(prob_sde, p = [cal_points[t]..., 0.1, 0.1])
        newdata = solve(new_prob, SOSRI(), saveat=0.1)
        # (model|new data)
        mod_newdata = fitlv_sde(newdata, prob_sde)
        # mcmc(model|new data)
        approx_samples_newdata = sample(mod_newdata, NUTS(0.25), N_samples; init_params=intial_all, progress=false, verbose = false)

        # samples from vi(model|new data)
        tr_approx_samples_newx[:,:,t] = hcat(bij.(getsamples(approx_samples_newdata, :pars))...)
            
        ProgressMeter.next!(pmeter)
    end
    ProgressMeter.finish!(pmeter)
end

if !use_cache & false # toggle to overwrite
    save_object("lotka-volterra-cache.jld2", 
        (data = data,
        true_p = true_p, 
        cal_p = cal_points, 
        tr_cal_s = tr_approx_samples_newx))
end


# resize
if !use_cache
    tr_approx_samples_newx = [tr_approx_samples_newx[:,i,j] for i in 1:N_energy, j in 1:N_importance]
end

tr_cal_points = bij.(cal_points)

# approximate weights
is_weights = ones(N_importance)

cal = Calibration(tr_cal_points, tr_approx_samples_newx)

d = BayesScoreCal.dimension(cal)[1]
tf = CholeskyAffine(d)
M = inv(Diagonal(std(cal.μs)))
res = energyscorecalibrate!(tf, cal, is_weights, scaling = M, penalty = (0.0, 0.05))

calcheck_approx = coverage(cal, checkprobs)
calcheck_adjust = coverage(cal, tf, checkprobs)

rmse(cal)
rmse(cal,tf)


approx_samples = vec(getsamples(ch_approx, :pars))
tr_approx_samples = bij.(approx_samples)
tf_samples = inv(bij).(tf.(tr_approx_samples, [mean(tr_approx_samples)]))

[mean(tf_samples) std(tf_samples)]

parnames = [Symbol("beta$i") for i in 1:4]

samples = DataFrame[]
push!(samples, 
    DataFrame(
        [parnames[i] => getindex.(approx_samples, i) for i in 1:4]...,
        :method => "Approx-post",
        :alpha => -1.0
    )
)

push!(samples, 
    DataFrame(
        [parnames[i] => getindex.(tf_samples, i) for i in 1:4]...,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)

push!(samples, 
    DataFrame(
        [parnames[i] => true_p[i] for i in 1:4]...,
        :method => "True-vals",
        :alpha => -1.0
    )
)

check = DataFrame[]

push!(check,
    DataFrame(
        [parnames[i] => getindex.(calcheck_approx, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Approx-post",
        :alpha => -1.0
    )
)

push!(check,
    DataFrame(
        [parnames[i] => getindex.(calcheck_adjust, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)

