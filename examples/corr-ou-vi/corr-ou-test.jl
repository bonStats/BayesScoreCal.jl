# Bivariate correlated OU process
# Approx model: Mean-field VI
# Approx samples: Direct via Bijectors.jl
# True model samples: MCMC via Turing.jl  

using Random
using Turing
using Turing: Variational
using DataFrames
import LinearAlgebra: PosDefException
using Bijectors
using Optim
using BayesScoreCal

# helper
getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))
dropvec(x::Vector{T}) where {T} = length(x) == 1 ? x[1] : x
paramindices(x::Symbol, indi::AbstractVector{Int64}) = length(indi) == 1 ? String(x) : ["$(String(x))[$i]" for i in indi]

# vector random generation
vrand(dist,N) = [rand(dist) for i in 1:N]

## setup data
T = 1.0
N = 100

# generate settings data
const x₀= [5.0, 5.0]
const μ = 1.0
const γ = 2.0
const ρ = 0.5
const σ = sqrt(20)   
const D = σ^2 / 2

const truevals = Dict([:μ => μ, :D => D, :ρ => ρ])

# approximate/true model settings
N_samples = 3000
const n_adapt = 1000

# optimisation settings
N_importance = 100
const N_energy = 2000
const energyβ = 1.0
vmultiplier = 2.0
alphalevels = [0.0, 0.25, 0.5, 0.9, 1.0]

calcheckprobs = range(0.1,0.95,step=0.05)

reps = 100

function cov2d(σ²,ρ)
    pcov = repeat([σ²], 2, 2) 
    pcov[[2,3]] .*= ρ
    return pcov
end

function ouprocess(T::Float64, x₀::Vector{Float64}, μ::Float64, γ::Float64, σ::Float64, ρ::Float64)
    pmean = μ .+ (x₀ .- μ) .* exp(-γ * T)
    σ² = (σ^2) * (1 - exp(-2*γ*T)) / (2 * γ)
    MvNormal(pmean, cov2d(σ², ρ))
end

ou = ouprocess(T, x₀, μ, γ, σ, ρ)

@model function model(x,x₀)
    D ~ Exponential(10) # scale = 10
    μ ~ Normal(0.0, 10.0)
    ρ ~ Uniform(0,1)
    obsμ = μ .+ (x₀ .- μ) .* exp(-γ * T)
    σ² = D * (1 - exp(-2*γ*T)) / γ
    for i in eachindex(x)
        x[i] ~ MvNormal(obsμ, cov2d(σ², ρ))
    end
end


function testfun(ou::MvNormal, T::Float64, N::Int64, N_importance::Int64, N_samples::Int64, vmultiplier::Float64, alphalevels::AbstractVector{Float64}, checkprobs::AbstractVector{Float64}, iter::Int64; options::Optim.Options = Optim.Options())

    dfsamples  = DataFrame[]
    dfcalcheck = DataFrame[]

    # X_T generate true data
    x = vrand(ou, N)

    # Instantiate model
    mod = model(x, x₀)

    pars = getparams(mod)
    par2tuple(x::Vector{Float64}) = NamedTuple(pars .=> x)

    Ndim = length(pars)
    Ndimchol = Int((Ndim^2 + Ndim)/2)

    parid(x::Symbol) = findfirst(pars .== x)

    # "true" posterior
    true_samples = PosDefException(1)
    while typeof(true_samples) <: Exception
        true_samples = try
            sample(mod, NUTS(n_adapt, 0.65), N_samples)
        catch err
            println("True posterior error")
            err
        end
    end

    # approximate posterior
    advi = ADVI(10, 1000)
    q_approx = vi(mod, advi)
    approx_samples = vrand(q_approx, N_samples)
    tr_approx_samples = inverse(q_approx.transform).(approx_samples)

    # calibration posterior (scale by vmultiplier on underlying transformed space)
    q_cal = transformed(
        Turing.TuringDiagMvNormal(q_approx.dist.m, q_approx.dist.σ * vmultiplier),
        q_approx.transform
    )
    
    # sample calibration points
    cal_points = vrand(q_cal, N_importance)
    tr_cal_points = inverse(q_cal.transform).(cal_points)

    # log prior evaluated on importance distribution samples
    ℓprior = [logprior(mod, par2tuple(v)) for (i, v) in enumerate(cal_points)]

    # log density evaluated on importance distribution samples (inverse transform and transform cancel)
    ℓcal = [logpdf(q_cal, v) for (i, v) in enumerate(cal_points)]

    # IS weights, jacobian of log transform in numerator and denominator cancel
    is_weights = ℓprior .- ℓcal

    # new data generation
    ou_given_cal = [ouprocess(T, x₀, v[parid(:μ)], γ, sqrt(2*v[parid(:D)]), v[parid(:ρ)]) for v in cal_points]

    # newx approx models: pre-allocate
    tr_approx_samples_newx = Matrix{typeof(cal_points[1])}(undef, N_energy, N_importance)

    for (t, modt) in enumerate(ou_given_cal) #

        # new data
        newxt = vrand(modt, N)
        # (model|new data)
        mod_newx = model(newxt, x₀)
        # vi(model|new data)
        q_newx = vi(mod_newx, advi)
        # samples from vi(model|new data)
        tr_approx_samples_newx[:,t] = vrand(q_newx.dist, N_energy)
        
    end

    # true and approx
    samplecomp = DataFrame(
        [p => vec(true_samples[p]) for p in pars]...,
        :method => "True-post",
        :iter => iter,
        :alpha => -1.0
    )

    push!(dfsamples, samplecomp)

    samplecomp = DataFrame(
        [p => getindex.(approx_samples, parid(p)) for p in pars]...,
        :method => "Approx-post",
        :iter => iter,
        :alpha => -1.0
    )

    push!(dfsamples, samplecomp)
    
    print("iter = $iter")

    # calibration object
    cal = Calibration(tr_cal_points, tr_approx_samples_newx)

    # approx cal
    for alpha in alphalevels

        trimval = quantile(is_weights, 1 - alpha)
        w = [is_weights[i] > trimval ? trimval : is_weights[i] for i in eachindex(is_weights)]
        w = exp.(w .- maximum(w))

        tf = CholeskyAffine(Ndim)
        
        # updates tf
        res = energyscorecalibrate!(tf, cal, w; β = energyβ, options = options)

        newsamples = q_approx.transform.( tf.(tr_approx_samples, [mean(tr_approx_samples)]) )

        samplecomp = DataFrame(
            [p => getindex.(newsamples, parid(p)) for p in pars]...,
            :method => "Adjust-post",
            :iter => iter,
            :alpha => alpha
        )

        push!(dfsamples, samplecomp)

        # Check calibration
        calcheck = coverage(cal, tf, checkprobs)

        calchecki = DataFrame(
            [p => getindex.(calcheck, parid(p)) for p in pars]...,
            :prob => checkprobs,
            :iter => iter,
            :alpha => alpha
        )

        push!(dfcalcheck, calchecki)

            # change to dataframe for output
    end

    return (samples = dfsamples, check = dfcalcheck)
    
end

dftruevals = DataFrame(
        truevals...,
        :method => "True-vals",
        :iter => 0,
        :alpha => -1.0
)


rr = testfun(ou, T, N, N_importance, N_samples, vmultiplier, alphalevels, calcheckprobs, 1, options = Optim.Options(f_tol = 0.00001))

allres = Vector{typeof(rr)}(undef, reps)

Threads.@threads for i in 1:reps
    allres[i] = testfun(ou, T, N, N_importance, N_samples, vmultiplier, alphalevels, calcheckprobs, i, options = Optim.Options(f_tol = 0.00001))
end

# summaries
ssamples = vcat(reduce(vcat, [getfield.(allres, :samples); dftruevals])...)
scalchecks = vcat(reduce(vcat,getfield.(allres, :check))...)
