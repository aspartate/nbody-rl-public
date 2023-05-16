mutable struct StandardScaler
    means::Vector{Float64}
    stds::Vector{Float64}
end

function fit!(scaler::StandardScaler, X::AbstractMatrix)
    scaler.means = mean(X, dims=1)[:]
    scaler.stds = std(X, dims=1)[:]
end

function transform(scaler::StandardScaler, X::AbstractMatrix)
    return (X .- scaler.means') ./ max.(scaler.stds', 1e-7) # Avoid division by zero
end

function inverse_transform(scaler::StandardScaler, X::AbstractMatrix)
    return (X .* scaler.stds') .+ scaler.means'
end

function build_scaler_nbody(env, num_episodes::Int64, initial_state = nothing)

    println("Building scaler...")
    
    # Make a copy of the environment
    env_ = deepcopy(env)

    # Initialize list of typical observations
    typical_observations = []
    typical_state_errors = []
    typical_energy_errors = []
    typical_rewards = []
    for _ in 1:num_episodes
        new_system!(env_, initial_state)
        done = false
        while !done
            flattened_trace, state_error, energy_error, state_reward, energy_reward, total_reward, done = env_step!(env_, rand(1:length(env.action_space)))
            # Append flattened_trace to typical_observations
            push!(typical_observations, flattened_trace)
            push!(typical_state_errors, state_error / env_.state_tol)
            push!(typical_energy_errors, energy_error / env_.energy_tol)
            push!(typical_rewards, total_reward)
        end
    end

    scaler = StandardScaler(Vector{Float64}(), Vector{Float64}())
    typical_observations = hcat(typical_observations...)' # Transpose to get shape (n_samples, n_features)
    fit!(scaler, typical_observations)

    println("Scaler built!")

    # Scatterplot of rewards vs. typical errors
    # scatter(typical_state_errors, typical_rewards, xlabel="Normalized Error/Tol", ylabel="Total Reward", label="State", legend=true, color=:green)
    # scatter!(typical_energy_errors, typical_rewards, label="Energy", legend=true, color=:purple)
    # vline!([1.0], color=:red, linestyle=:dash, label="Error/Tol = 1")
    # display(current())

    return scaler
end

function build_scaler_lorenz(env, num_episodes::Int64, initial_state = nothing)

    println("Building scaler...")
    
    # Make a copy of the environment
    env_ = deepcopy(env)

    # Initialize list of typical observations
    typical_observations = []
    typical_errors = []
    typical_rewards = []
    for _ in 1:num_episodes
        new_system!(env_, initial_state)
        done = false
        while !done
            flattened_trace, error, reward, done = env_step!(env_, rand(1:length(env.action_space)))
            # Append flattened_trace to typical_observations
            push!(typical_observations, flattened_trace)
            push!(typical_errors, error / env_.tol)
            push!(typical_rewards, reward)
        end
    end

    scaler = StandardScaler(Vector{Float64}(), Vector{Float64}())
    typical_observations = hcat(typical_observations...)' # Transpose to get shape (n_samples, n_features)
    fit!(scaler, typical_observations)

    println("Scaler built!")

    # # Scatterplot of rewards vs. typical errors
    # scatter(typical_errors, typical_rewards, xlabel="Normalized Error/Tol", ylabel="Total Reward", label="State", legend=true, color=:green)
    # vline!([1.0], color=:red, linestyle=:dash, label="Error/Tol = 1")
    # display(current())

    return scaler
end