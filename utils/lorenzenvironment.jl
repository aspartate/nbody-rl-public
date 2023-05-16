mutable struct LorenzEnvironment
    action_space::Vector{Float64}
    tol::Float64
    pos_reward_range::Tuple{Float64,Float64}
    max_time::Float64
    method::String
    systems::Vector{LorenzSystem}
end

function buildLorenzEnvironment(
    action_space::Vector{Float64},
    tol::Float64,
    pos_reward_range::Tuple{Float64,Float64},
    max_time::Float64,
    method::String,
)
    return LorenzEnvironment(action_space, tol, pos_reward_range, max_time, method, [])
end

function env_reward(env::LorenzEnvironment, step_size::Float64, error::Float64)
    a = (env.pos_reward_range[1] - env.pos_reward_range[2]) / log(env.action_space[1] / env.action_space[end])
    b = exp(env.pos_reward_range[2] / a) / env.action_space[end]

    if all(error .<= env.tol)
        return a * log(b * step_size)
    else
        return log10(env.tol / error)
    end
end

function env_step!(env::LorenzEnvironment, action::Int)
    estimated_next_state = system_step!(env.systems[end], env.action_space[action])
    true_next_state = system_gt!(env.systems[end])
    error = norm(estimated_next_state - true_next_state)
    r = env_reward(env, env.action_space[action], error)
    done = env.systems[end].time[end] >= env.max_time
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace, error, r, done
end

function env_step_evalmode!(env::LorenzEnvironment, action::Int)
    # Same as env_step, but doesn't return error or reward because speed is of the essence
    estimated_next_state = system_step!(env.systems[end], env.action_space[action])
    done = env.systems[end].time[end] >= env.max_time
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace, done
end

function get_random_initial_state(env::LorenzEnvironment)
    position = [rand(-10.0:0.1:10.0), rand(-10.0:0.1:10.0), rand(15.0:0.1:35.0)]
    return position
end

function new_system!(env::LorenzEnvironment, initial_state=nothing)
    if initial_state === nothing
        initial_state = get_random_initial_state(env)
    end
    push!(env.systems, buildLorenzSystem(initial_state, env.method))
    system_step!(env.systems[end], env.action_space[1]) # Take an initial step to generate starting eval trace
    system_gt!(env.systems[end]) # Needed to make eval trace and true state the same length
    flattened_trace = vcat([vcat(env.systems[end].eval_trace[end].k...), env.systems[end].eval_trace[end].dt]...)
    return flattened_trace
end