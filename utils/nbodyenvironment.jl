mutable struct NBodyEnvironment
    action_space::Vector{Float64}
    state_tol::Float64
    energy_tol::Float64
    alpha::Float64
    pos_reward_range::Tuple{Float64,Float64}
    max_time::Float64
    method::String
    n_bodies::Int
    gravity::Float64
    systems::Vector{NBodySystem}
end

function buildNBodyEnvironment(
    action_space::Vector{Float64},
    state_tol::Float64,
    energy_tol::Float64,
    alpha::Float64,
    pos_reward_range::Tuple{Float64,Float64},
    max_time::Float64,
    method::String,
    n_bodies::Int,
    gravity::Float64
)
    return NBodyEnvironment(action_space, state_tol, energy_tol, alpha, pos_reward_range, max_time, method, n_bodies, gravity, [])
end

function env_reward(env::NBodyEnvironment, step_size::Float64, error::Float64, tol::Float64)
    a = (env.pos_reward_range[1] - env.pos_reward_range[2]) / log(env.action_space[1] / env.action_space[end])
    b = exp(env.pos_reward_range[2] / a) / env.action_space[end]

    if all(error .<= tol)
        return a * log(b * step_size)
    else
        return log10(tol / error)
    end
end

function env_step!(env::NBodyEnvironment, action::Int)
    estimated_next_state = system_step!(env.systems[end], env.action_space[action])
    true_next_state = system_gt!(env.systems[end])
    state_error = norm(estimated_next_state - true_next_state) # / norm(true_next_state)                                            # State error
    energy_error = abs(env.systems[end].energy[end] - env.systems[end].energy[end-1]) / abs(env.systems[end].energy[end-1]) # Energy error
    state_reward = env_reward(env, env.action_space[action], state_error, env.state_tol)               # State reward
    energy_reward = env_reward(env, env.action_space[action], energy_error, env.energy_tol)             # Energy reward
    # println("State error: ", state_error)
    # println("Energy error: ", energy_error)
    # println("State reward: ", state_reward)
    # println("Energy reward: ", energy_reward)
    total_reward = env.alpha * state_reward + (1 - env.alpha) * energy_reward                      # Total reward
    done = env.systems[end].time[end] >= env.max_time
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), vcat(env.systems[end].eval_trace[end].masses...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace, state_error, energy_error, state_reward, energy_reward, total_reward, done
end

function env_step_evalmode!(env::NBodyEnvironment, action::Int)
    # Same as env_step, but doesn't return error or reward because speed is of the essence
    estimated_next_state = system_step!(env.systems[end], env.action_space[action])
    done = env.systems[end].time[end] >= env.max_time
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), vcat(env.systems[end].eval_trace[end].masses...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace, done
end

function get_random_initial_state(env::NBodyEnvironment)
    bodies = Body[]
    for i in 1:env.n_bodies
        mass = rand() * 10
        position = [rand() * 20 - 10, rand() * 20 - 10, rand() * 20 - 10]
        velocity = [rand() * 20 - 10, rand() * 20 - 10, rand() * 20 - 10]
        acceleration = [0.0, 0.0, 0.0]
        push!(bodies, Body(mass, position, velocity, acceleration))
    end
    return bodies
end

function new_system!(env::NBodyEnvironment, initial_state=nothing)
    if initial_state === nothing
        initial_state = get_random_initial_state(env)
    end
    push!(env.systems, buildNBodySystem(initial_state, env.gravity, env.method))
    system_step!(env.systems[end], env.action_space[1]) # Take an initial step to generate starting eval trace
    system_gt!(env.systems[end]) # Needed to make eval trace and true state the same length
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), vcat(env.systems[end].eval_trace[end].masses...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace
end

function estimate_error_bounds(env::NBodyEnvironment)

    # Get avg min error
    env_min = deepcopy(env)
    min_state_errors = []
    min_energy_errors = []
    done = false
    while !done
        action = 1
        next_observation, state_error, energy_error, state_reward, energy_reward, total_reward, done = env_step!(env_min, action)
        push!(min_state_errors, state_error)
        push!(min_energy_errors, energy_error)
    end
    avg_min_state_error = mean(min_state_errors)
    avg_min_energy_error = mean(min_energy_errors)

    # Get avg max error
    env_max = deepcopy(env)
    max_state_errors = []
    max_energy_errors = []
    done = false
    while !done
        action = length(env_max.action_space)
        next_observation, state_error, energy_error, state_reward, energy_reward, total_reward, done = env_step!(env_max, action)
        push!(max_state_errors, state_error)
        push!(max_energy_errors, energy_error)
    end
    avg_max_state_error = mean(max_state_errors)
    avg_max_energy_error = mean(max_energy_errors)


    return avg_min_state_error, avg_max_state_error, avg_min_energy_error, avg_max_energy_error
end