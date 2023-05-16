using DifferentialEquations
using LinearAlgebra
using Plots


struct EvalTrace
    k::Vector{Any}
    masses::Vector{Float64}
    dt::Float64
end

struct Body
    mass::Float64
    position::Array{Float64,1}
    velocity::Array{Float64,1}
    acceleration::Array{Float64,1}
end

mutable struct NBodySystem
    true_state::Vector{Array{Float64}}
    estimated_state::Vector{Array{Float64}}
    eval_trace::Vector{EvalTrace}
    time::Vector{Float64}
    method::String
    masses::Vector{Float64}
    gravity::Float64
    energy::Vector{Float64}
end


function mutate(body::Body, scale::Float64)::Body
    # Returns mutated body by randomly perturbing mass, position, velocity, and acceleration
    mass = body.mass * abs(1 + randn() * scale)
    position = body.position .* (1 .+ randn(3) * scale)
    velocity = body.velocity
    acceleration = body.acceleration
    Body(mass, position, velocity, acceleration)
end

function rotate(body::Body)::Body
    # Returns rotated body by randomly rotating position around origin
    theta = rand() * 2 * pi
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]
    new_position = R * body.position

    # Calculate new velocity and acceleration
    new_velocity = R * body.velocity
    new_acceleration = R * body.acceleration

    Body(body.mass, new_position, new_velocity, new_acceleration)
end


function getEnergy(bodies::Vector{Body}, gravity::Float64)
    # Returns total energy of system (Hamiltonian estimator)
    energy = 0.0
    for i in 1:length(bodies)
        body = bodies[i]
        energy += 0.5 * body.mass * norm(body.velocity)^2
        for j in 1:length(bodies)
            if i != j
                energy -= gravity * body.mass * bodies[j].mass / norm(body.position - bodies[j].position)
            end
        end
    end
    return energy
end

function get_energy_from_state(state::Array{Float64}, system::NBodySystem)
    num_bodies = Int64(length(state) / 6)
    positions = state[1:3*num_bodies]
    velocities = state[3*num_bodies+1:end]
    energy = 0.0
    for i in 1:num_bodies
        energy += 0.5 * system.masses[i] * norm(velocities[3*(i-1)+1:3*i])^2
        for j in 1:num_bodies
            if i != j
                energy -= system.gravity * system.masses[i] * system.masses[j] / norm(positions[3*(i-1)+1:3*i] - positions[3*(j-1)+1:3*j])
            end
        end
    end
    return energy
end

function getBodies(system::NBodySystem)
    # Returns bodies from system using the last estimated state
    num_bodies = length(system.masses)
    bodies = Body[]
    for i in 1:num_bodies
        push!(bodies, Body(system.masses[i], system.estimated_state[end][3*(i-1)+1:3*i], system.estimated_state[end][3*num_bodies+3*(i-1)+1:3*num_bodies+3*i], zeros(3)))
    end
    return bodies
end


function plotEnergy(system::NBodySystem)
    # Plots energy of system over time
    display(plot(system.time, system.energy, xlabel="Time", ylabel="Energy", size=(600, 200), label=false))
end


function plotRelativeEnergyError(system::NBodySystem)
    # Plots relative energy error of system over time
    display(plot(system.time, (system.energy .- system.energy[1]) ./ system.energy[1], xlabel="Time", ylabel="Relative Energy Error", size=(600, 200), label=false))
end

function buildNBodySystem(bodies::Vector{Body}, gravity::Float64, method::String)
    # initial_state is a flattened vector of all positions, followed by all velocities
    initial_state = vcat([body.position for body in bodies]..., [body.velocity for body in bodies]...)
    masses = [body.mass for body in bodies]
    initial_energy = getEnergy(bodies, gravity)
    return NBodySystem([initial_state], [initial_state], [], [0.0], method, masses, gravity, [initial_energy])
end

function function_(time::Float64, state::Array{Float64,1}, system::NBodySystem, padding::Float64=0.0)
    num_bodies = length(system.masses)
    @assert length(state) == 6 * num_bodies "Expected state vector of length $(6 * num_bodies), got length $(length(state))"
    positions = state[1:3*num_bodies]
    velocities = state[3*num_bodies+1:end]
    accelerations = zeros(3 * num_bodies)

    for i in 1:num_bodies
        for j in 1:num_bodies
            if i != j
                r = positions[3*(j-1)+1:3*j] - positions[3*(i-1)+1:3*i]
                accelerations[3*(i-1)+1:3*i] += system.gravity * system.masses[j] * r / (norm(r)^3 + system.masses[j] * padding)
            end
        end
    end
    derivatives = vcat(velocities, accelerations)
    return derivatives
end

function system_step!(system::NBodySystem, dt::Float64, padding::Float64=0.0)
    prev_state = system.estimated_state[end]
    if system.method == "RK4"
        k1 = function_(system.time[end], prev_state, system, padding)
        k2 = function_(system.time[end] + dt / 2, prev_state + dt / 2 * k1, system, padding)
        k3 = function_(system.time[end] + dt / 2, prev_state + dt / 2 * k2, system, padding)
        k4 = function_(system.time[end] + dt, prev_state + dt * k3, system, padding)
        push!(system.eval_trace, EvalTrace([k1, k2, k3, k4], system.masses, dt))
        deltas = (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * dt
    elseif system.method == "RKDP"
        k1 = function_(system.time[end], prev_state, system, padding)
        k2 = function_(system.time[end] + (1 / 5) * dt, prev_state + dt * (1 / 5) * k1, system, padding)
        k3 = function_(system.time[end] + (3 / 10) * dt, prev_state + dt * (3 / 40) * k1 + dt * (9 / 40) * k2, system, padding)
        k4 = function_(system.time[end] + (4 / 5) * dt, prev_state + dt * (44 / 45) * k1 - dt * (56 / 15) * k2 + dt * (32 / 9) * k3, system, padding)
        k5 = function_(system.time[end] + dt, prev_state + dt * (19372 / 6561) * k1 - dt * (25360 / 2187) * k2 + dt * (64448 / 6561) * k3 - dt * (212 / 729) * k4, system, padding)
        k6 = function_(system.time[end] + dt, prev_state + dt * (9017 / 3168) * k1 - dt * (355 / 33) * k2 + dt * (46732 / 5247) * k3 + dt * (49 / 176) * k4 - dt * (5103 / 18656) * k5, system, padding)
        push!(system.eval_trace, EvalTrace([k1, k2, k3, k4, k5, k6], system.masses, dt))
        deltas = (k1 * (35 / 384) + k3 * (500 / 1113) + k4 * (125 / 192) - k5 * (2187 / 6784) + k6 * (11 / 84)) * dt
    else
        error("Method not implemented")
    end
    push!(system.estimated_state, prev_state + deltas) # Update current state with estimated solution
    push!(system.time, system.time[end] + dt) # Update time
    push!(system.energy, getEnergy(getBodies(system), system.gravity)) # Update energy
    return system.estimated_state[end]
end

function system_gt!(system::NBodySystem, timespan=nothing, initial_state=nothing)
    if timespan === nothing
        timespan = (system.time[end-1], system.time[end])
    end
    if initial_state === nothing
        initial_state = system.estimated_state[end-1]
    end
    # Make function compatible with DifferentialEquations.jl
    ODEFunctionWrapper = (u, p, t) -> function_(t, u, system)
    prob = ODEProblem(ODEFunctionWrapper, initial_state, timespan)
    sol = solve(prob, DP5(), reltol=1e-8)
    push!(system.true_state, sol.u[end])
    return system.true_state[end]
end

function plot_trajectory(system::NBodySystem, states=nothing, trail=200)
    num_bodies = length(system.masses)
    if states === nothing
        states = hcat(system.estimated_state...)
    end

    # Define a list of colors for the trajectories
    colors = [:red, :green, :blue, :yellow, :black, :orange, :purple, :pink, :cyan, :magenta]

    max_val = 0.0
    for i in 1:num_bodies
        x_values = abs.(states[3*(i-1)+1, :])
        y_values = abs.(states[3*(i-1)+2, :])
        z_values = abs.(states[3*(i-1)+3, :])
        max_val = max(max_val, maximum(x_values), maximum(y_values), maximum(z_values))
    end

    function plot_single_state(state_i::Int64=-1, trail::Int64=200)
        # plot the state at index i
        plot3d()

        for i in 1:num_bodies
            # Get the color for this body from the color list, cycling if necessary
            color = colors[mod(i - 1, length(colors))+1]
            plot3d!(
                states[3*(i-1)+1, max(1, state_i - trail):state_i],
                states[3*(i-1)+2, max(1, state_i - trail):state_i],
                states[3*(i-1)+3, max(1, state_i - trail):state_i],
                color=color,
                label=false
            )
            scatter3d!(
                [states[3*(i-1)+1, state_i]],
                [states[3*(i-1)+2, state_i]],
                [states[3*(i-1)+3, state_i]],
                color=color,
                label="Body $i",
                markersize=5,
                legend=:topright
            )
        end
        xlims!(-max_val, max_val)
        ylims!(-max_val, max_val)
        zlims!(-max_val, max_val)
        xlabel!("x")
        ylabel!("y")
        zlabel!("z")
    end

    anim = @animate for i in 1:size(states, 2)
        plot_single_state(i, trail)
    end
    return anim

    # if show
    #     display(current())
    # end
end



function center_initial_state(state::Vector{Body})
    # calculate center of mass
    center_of_mass = [0.0, 0.0, 0.0]
    total_mass = 0.0
    for body in state
        center_of_mass += body.mass * body.position
        total_mass += body.mass
    end
    center_of_mass ./= total_mass

    # calculate center of mass velocity
    center_of_mass_vel = [0.0, 0.0, 0.0]
    for body in state
        center_of_mass_vel += body.mass * body.velocity
    end
    center_of_mass_vel ./= total_mass

    new_state = Body[]
    for body in state
        push!(new_state, Body(body.mass, body.position .- center_of_mass, body.velocity .- center_of_mass_vel, body.acceleration))
    end
    return new_state
end
