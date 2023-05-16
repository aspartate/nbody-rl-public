using DifferentialEquations
using LinearAlgebra
using Plots


struct EvalTrace
    k::Vector{Any}
    dt::Float64
end

mutable struct LorenzSystem
    true_state::Vector{Array{Float64}}
    estimated_state::Vector{Array{Float64}}
    eval_trace::Vector{EvalTrace}
    time::Vector{Float64}
    method::String
end

function buildLorenzSystem(initial_state, method)
    system = LorenzSystem([initial_state], [initial_state], [], [0.0], method)
    @assert method in ["RK4", "RKDP"]
    return system
end

function function_(time, state)
    # Returns derivatives of state vector at given time
    sigma = 10
    rho = 28
    beta = 8 / 3
    derivatives = [
        sigma * (state[2] - state[1]),
        state[1] * (rho - state[3]) - state[2],
        state[1] * state[2] - beta * state[3],
    ]
    return derivatives
end

function system_step!(system::LorenzSystem, dt::Float64)
    prev_state = system.estimated_state[end]
    if system.method == "RK4"
        k1 = function_(system.time[end], prev_state)
        k2 = function_(system.time[end] + dt / 2, prev_state + dt / 2 * k1)
        k3 = function_(system.time[end] + dt / 2, prev_state + dt / 2 * k2)
        k4 = function_(system.time[end] + dt, prev_state + dt * k3)
        push!(system.eval_trace, EvalTrace([k1, k2, k3, k4], dt))
        deltas = (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * dt
    elseif system.method == "RKDP"
        k1 = function_(system.time[end], prev_state)
        k2 = function_(system.time[end] + (1 / 5) * dt, prev_state + dt * (1 / 5) * k1)
        k3 = function_(system.time[end] + (3 / 10) * dt, prev_state + dt * (3 / 40) * k1 + dt * (9 / 40) * k2)
        k4 = function_(system.time[end] + (4 / 5) * dt, prev_state + dt * (44 / 45) * k1 - dt * (56 / 15) * k2 + dt * (32 / 9) * k3)
        k5 = function_(system.time[end] + dt, prev_state + dt * (19372 / 6561) * k1 - dt * (25360 / 2187) * k2 + dt * (64448 / 6561) * k3 - dt * (212 / 729) * k4)
        k6 = function_(system.time[end] + dt, prev_state + dt * (9017 / 3168) * k1 - dt * (355 / 33) * k2 + dt * (46732 / 5247) * k3 + dt * (49 / 176) * k4 - dt * (5103 / 18656) * k5)
        push!(system.eval_trace, EvalTrace([k1, k2, k3, k4, k5, k6], dt))
        deltas = (k1 * (35 / 384) + k3 * (500 / 1113) + k4 * (125 / 192) - k5 * (2187 / 6784) + k6 * (11 / 84)) * dt
    else
        error("Method not implemented")
    end
    push!(system.estimated_state, prev_state + deltas) # Update current state with estimated solution
    push!(system.time, system.time[end] + dt) # Update time
    return system.estimated_state[end]
end

function system_gt!(system::LorenzSystem, timespan=nothing, initial_state=nothing)
    # Returns ground truth solution at current time, using RK45 method
    # Analogy: Agent is taking big steps down a hill, whereas ground truth is a ball rolling down a hill
    if timespan === nothing
        timespan = (system.time[end-1], system.time[end]) # Integrate from previous time to current time
    end
    if initial_state === nothing
        initial_state = system.estimated_state[end-1] # Integrate from previous estimated state
    end
    # Make function compatible with DifferentialEquations.jl
    ODEFunctionWrapper = (u, p, t) -> function_(t, u)
    prob = ODEProblem(ODEFunctionWrapper, initial_state, timespan)
    sol = solve(prob, DP5(), reltol=1e-8)
    push!(system.true_state, sol.u[end]) # Update true state with ground truth solution
    return system.true_state[end]
end

function plot_trajectory(system::LorenzSystem, states=nothing, show=true)
    if states === nothing
        states = hcat(system.estimated_state...)
    end
    plot3d()
    color = :jet
    plot3d!(
        states[1, :],
        states[2, :],
        states[3, :],
        color=color,
        legend=:bottomright
    )
    scatter3d!(
        [states[1, end]],
        [states[2, end]],
        [states[3, end]],
        color=color,
        markersize=5
    )
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")

    if show
        display(current())
    end
end